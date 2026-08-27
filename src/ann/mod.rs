//! Approximate nearest neighbour search over DINOv2 embeddings.
//!
//! Recognition used to compare every embedding with every other one. At ten
//! thousand photos that is 49,995,000 cosine computations; at fifty thousand it
//! is 1,249,975,000. This module replaces that sweep with a hierarchical
//! navigable small world graph: each photo is queried for its nearest `top_k`
//! neighbours, and only those pairs go on to be classified.
//!
//! Two things this module is deliberately *not*:
//!
//! - It is not a similarity decision. It only proposes candidates. The caller
//!   still recomputes an exact cosine and runs the full `classify_pair` gate,
//!   so a neighbour returned here is not thereby a duplicate.
//! - It is not on the exact-duplicate path. Byte-identical files are found by
//!   SHA-256 and never depend on being inside anyone's `top_k`.
//!
//! The graph is built in-process with no new dependency. An HNSW crate would
//! have worked too, but PhotoCleaner ships as a portable Windows folder and the
//! algorithm is small enough that owning it costs less than owning a
//! dependency's build requirements.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use anyhow::{bail, Result};

/// Below this many embeddings the brute-force sweep is cheaper than building a
/// graph: 499 vectors is at most 124,251 pairs, which is nothing.
pub const ANN_MIN_LIBRARY_SIZE: usize = 500;

/// Neighbours requested per photo. Generous on purpose - the classifier does
/// the real filtering, and a burst of near-identical frames can easily fill a
/// dozen slots on its own.
pub const ANN_TOP_K: usize = 32;

/// Bumped when the graph's construction parameters change in a way that would
/// invalidate a persisted index. Nothing persists an index yet; the descriptor
/// exists so that adding one later does not need a schema discussion.
pub const ANN_INDEX_VERSION: u32 = 1;

/// Outgoing edges kept per node above layer 0.
const HNSW_M: usize = 16;
/// Outgoing edges kept per node on layer 0, where the graph must stay dense
/// enough to be navigable.
const HNSW_M_MAX0: usize = 32;
/// Beam width during construction.
const HNSW_EF_CONSTRUCTION: usize = 100;
/// Floor on the beam width at query time.
const HNSW_EF_SEARCH_MIN: usize = 64;
/// Guards against a pathological level draw producing a near-empty tower.
const HNSW_MAX_LEVEL: usize = 16;

/// How the candidate pairs for a library were produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateSearchMode {
    /// Every pair was examined. Used below [`ANN_MIN_LIBRARY_SIZE`].
    BruteForce,
    /// Pairs came from the neighbour graph.
    Ann,
    /// The graph could not be built or queried, so the exhaustive sweep ran
    /// instead. Recognition stays correct; it just costs what it used to.
    BruteForceFallback,
}

impl CandidateSearchMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::BruteForce => "BRUTE_FORCE",
            Self::Ann => "ANN_HNSW",
            Self::BruteForceFallback => "BRUTE_FORCE_FALLBACK",
        }
    }
}

/// One proposed pair. `left` and `right` index into the slice handed to
/// [`search_candidates`], canonicalised so `left < right`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnnCandidate {
    pub left: usize,
    pub right: usize,
    /// Cosine as computed inside the index, in f32. The caller recomputes it in
    /// f64 before deciding anything; this value is for diagnostics and ordering.
    pub cosine: f32,
}

/// Identity of the vectors an index was built from.
///
/// Reserved for a persisted index: any of these changing means a stored graph
/// describes vectors that no longer exist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnnDescriptor {
    pub library_id: String,
    pub model_hash: Option<String>,
    pub dimension: usize,
    pub dtype: String,
    pub index_version: u32,
}

impl AnnDescriptor {
    pub fn new(library_id: &str, model_hash: Option<&str>, dimension: usize, dtype: &str) -> Self {
        Self {
            library_id: library_id.to_string(),
            model_hash: model_hash.map(str::to_string),
            dimension,
            dtype: dtype.to_string(),
            index_version: ANN_INDEX_VERSION,
        }
    }
}

/// Result of one library's candidate generation, including enough counters to
/// tell from a report whether the graph ran and how much it pruned.
#[derive(Clone, Debug)]
pub struct CandidateSearchOutcome {
    pub mode: CandidateSearchMode,
    pub candidates: Vec<AnnCandidate>,
    /// Number of neighbour queries issued (zero when brute force ran).
    pub queries: usize,
    /// Neighbours returned across all queries, before canonicalising.
    pub raw_neighbors: usize,
    /// Distinct pairs after canonicalising and de-duplicating.
    pub unique_pairs: usize,
}

/// Produces the candidate pairs for a set of embeddings.
///
/// Never returns an error: if the graph fails, the exhaustive sweep runs and the
/// mode says so. A library must not fail to scan because an index would not
/// build.
pub fn search_candidates(vectors: &[&[f32]], top_k: usize) -> CandidateSearchOutcome {
    let count = vectors.len();
    if count < 2 {
        return CandidateSearchOutcome {
            mode: CandidateSearchMode::BruteForce,
            candidates: Vec::new(),
            queries: 0,
            raw_neighbors: 0,
            unique_pairs: 0,
        };
    }
    if count < ANN_MIN_LIBRARY_SIZE {
        let candidates = brute_force_candidates(vectors);
        let unique_pairs = candidates.len();
        return CandidateSearchOutcome {
            mode: CandidateSearchMode::BruteForce,
            candidates,
            queries: 0,
            raw_neighbors: 0,
            unique_pairs,
        };
    }

    match HnswIndex::build(vectors) {
        Ok(index) => index.candidate_pairs(top_k.max(1)),
        Err(error) => {
            crate::logging::error(format!(
                "ANN index build failed, falling back to the exhaustive sweep: {error:#}"
            ));
            let candidates = brute_force_candidates(vectors);
            let unique_pairs = candidates.len();
            CandidateSearchOutcome {
                mode: CandidateSearchMode::BruteForceFallback,
                candidates,
                queries: 0,
                raw_neighbors: 0,
                unique_pairs,
            }
        }
    }
}

fn brute_force_candidates(vectors: &[&[f32]]) -> Vec<AnnCandidate> {
    let mut candidates = Vec::new();
    for left in 0..vectors.len() {
        for right in (left + 1)..vectors.len() {
            candidates.push(AnnCandidate {
                left,
                right,
                cosine: dot(vectors[left], vectors[right]),
            });
        }
    }
    candidates
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    let len = left.len().min(right.len());
    let mut sum = 0f32;
    for index in 0..len {
        sum += left[index] * right[index];
    }
    sum
}

fn normalized(vector: &[f32]) -> Vec<f32> {
    let norm = vector
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    if norm <= 0.0 {
        return vector.to_vec();
    }
    vector.iter().map(|v| (*v as f64 / norm) as f32).collect()
}

/// A node paired with its distance, ordered nearest-first.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Scored {
    distance: f32,
    node: u32,
}

impl Eq for Scored {}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        // NaN cannot arise from normalized finite vectors, but ordering must be
        // total for BinaryHeap regardless.
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Stamp-based visited set, allocated once and reused for every search.
///
/// A fresh `vec![false; n]` per query would itself be O(n) per query, which is
/// exactly the cost this module exists to remove.
struct VisitSet {
    stamps: Vec<u32>,
    epoch: u32,
}

impl VisitSet {
    fn new(len: usize) -> Self {
        Self {
            stamps: vec![0; len],
            epoch: 0,
        }
    }

    fn begin(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            for stamp in self.stamps.iter_mut() {
                *stamp = 0;
            }
            self.epoch = 1;
        }
    }

    /// Returns true when the node had already been seen this epoch.
    fn seen(&mut self, node: usize) -> bool {
        match self.stamps.get_mut(node) {
            Some(stamp) if *stamp == self.epoch => true,
            Some(stamp) => {
                *stamp = self.epoch;
                false
            }
            None => true,
        }
    }
}

/// Hierarchical navigable small world graph over unit vectors.
pub struct HnswIndex {
    vectors: Vec<Vec<f32>>,
    /// `neighbors[node][level]`. A node has `levels[node] + 1` levels.
    neighbors: Vec<Vec<Vec<u32>>>,
    levels: Vec<usize>,
    entry: Option<usize>,
    max_level: usize,
}

impl HnswIndex {
    pub fn build(vectors: &[&[f32]]) -> Result<Self> {
        if vectors.is_empty() {
            bail!("cannot build an ANN index over zero vectors");
        }
        let dimension = vectors[0].len();
        if dimension == 0 {
            bail!("cannot build an ANN index over zero-dimensional vectors");
        }
        if let Some(bad) = vectors.iter().find(|v| v.len() != dimension) {
            bail!(
                "inconsistent embedding dimension: expected {dimension}, found {}",
                bad.len()
            );
        }

        let level_scale = 1.0 / (HNSW_M as f64).ln().max(f64::MIN_POSITIVE);
        let levels: Vec<usize> = (0..vectors.len())
            .map(|node| level_for(node, level_scale))
            .collect();
        // Convert to owned unit vectors exactly once. Every later distance is a
        // dot product on these, never a fresh decode of the stored blob.
        let owned: Vec<Vec<f32>> = vectors.iter().map(|v| normalized(v)).collect();
        let neighbors: Vec<Vec<Vec<u32>>> = levels
            .iter()
            .map(|level| vec![Vec::new(); level + 1])
            .collect();

        let mut index = Self {
            vectors: owned,
            neighbors,
            levels,
            entry: None,
            max_level: 0,
        };
        let mut visit = VisitSet::new(index.vectors.len());
        for node in 0..index.vectors.len() {
            index.insert(node, &mut visit);
        }
        Ok(index)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    fn distance_to(&self, query: &[f32], node: usize) -> f32 {
        match self.vectors.get(node) {
            Some(vector) => 1.0 - dot(query, vector),
            None => f32::MAX,
        }
    }

    fn distance_between(&self, left: usize, right: usize) -> f32 {
        match (self.vectors.get(left), self.vectors.get(right)) {
            (Some(a), Some(b)) => 1.0 - dot(a, b),
            _ => f32::MAX,
        }
    }

    fn insert(&mut self, node: usize, visit: &mut VisitSet) {
        let level = self.levels[node];
        let Some(entry) = self.entry else {
            self.entry = Some(node);
            self.max_level = level;
            return;
        };

        let query = self.vectors[node].clone();
        let mut current = entry;

        // Greedy descent through the layers this node does not belong to.
        let mut layer = self.max_level;
        while layer > level {
            current = self.greedy_descend(&query, current, layer);
            layer -= 1;
        }

        let mut entry_points = vec![current];
        let top = level.min(self.max_level);
        for layer in (0..=top).rev() {
            let found =
                self.search_layer(&query, &entry_points, HNSW_EF_CONSTRUCTION, layer, visit);
            let capacity = if layer == 0 { HNSW_M_MAX0 } else { HNSW_M };
            let selected: Vec<u32> = found
                .iter()
                .filter(|scored| scored.node as usize != node)
                .take(capacity)
                .map(|scored| scored.node)
                .collect();

            self.neighbors[node][layer] = selected.clone();
            for neighbor in selected.iter().copied() {
                self.link_back(neighbor as usize, node, layer, capacity);
            }

            entry_points = if found.is_empty() {
                vec![current]
            } else {
                found.iter().map(|scored| scored.node as usize).collect()
            };
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(node);
        }
    }

    fn greedy_descend(&self, query: &[f32], start: usize, layer: usize) -> usize {
        let mut current = start;
        let mut best = self.distance_to(query, current);
        loop {
            let Some(edges) = self.neighbors.get(current).and_then(|n| n.get(layer)) else {
                return current;
            };
            let mut improved = None;
            for neighbor in edges.iter().copied() {
                let candidate = neighbor as usize;
                let distance = self.distance_to(query, candidate);
                if distance < best {
                    best = distance;
                    improved = Some(candidate);
                }
            }
            match improved {
                Some(next) => current = next,
                None => return current,
            }
        }
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        visit: &mut VisitSet,
    ) -> Vec<Scored> {
        let ef = ef.max(1);
        visit.begin();
        let mut frontier: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        let mut result: BinaryHeap<Scored> = BinaryHeap::new();

        for entry in entry_points.iter().copied() {
            if entry >= self.vectors.len() || visit.seen(entry) {
                continue;
            }
            let scored = Scored {
                distance: self.distance_to(query, entry),
                node: entry as u32,
            };
            frontier.push(Reverse(scored));
            result.push(scored);
        }
        while result.len() > ef {
            result.pop();
        }

        while let Some(Reverse(current)) = frontier.pop() {
            let worst = result.peek().map(|scored| scored.distance);
            if let Some(worst) = worst {
                if current.distance > worst && result.len() >= ef {
                    break;
                }
            }
            let node = current.node as usize;
            let Some(edges) = self.neighbors.get(node).and_then(|n| n.get(layer)) else {
                continue;
            };
            for neighbor in edges.iter().copied() {
                let candidate = neighbor as usize;
                if visit.seen(candidate) {
                    continue;
                }
                let scored = Scored {
                    distance: self.distance_to(query, candidate),
                    node: neighbor,
                };
                let worst = result.peek().map(|item| item.distance);
                let accept = result.len() < ef || worst.is_some_and(|w| scored.distance < w);
                if accept {
                    frontier.push(Reverse(scored));
                    result.push(scored);
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }

        result.into_sorted_vec()
    }

    fn link_back(&mut self, node: usize, new_neighbor: usize, layer: usize, capacity: usize) {
        let Some(existing) = self.neighbors.get(node).and_then(|n| n.get(layer)) else {
            return;
        };
        if existing.iter().any(|n| *n as usize == new_neighbor) {
            return;
        }
        let mut edges = existing.clone();
        edges.push(new_neighbor as u32);
        if edges.len() > capacity {
            let mut scored: Vec<Scored> = edges
                .iter()
                .map(|edge| Scored {
                    distance: self.distance_between(node, *edge as usize),
                    node: *edge,
                })
                .collect();
            scored.sort();
            scored.truncate(capacity);
            edges = scored.into_iter().map(|item| item.node).collect();
        }
        if let Some(slot) = self.neighbors.get_mut(node).and_then(|n| n.get_mut(layer)) {
            *slot = edges;
        }
    }

    /// Returns the `top_k` nearest neighbours of `node`, excluding itself.
    fn neighbors_of(&self, node: usize, top_k: usize, visit: &mut VisitSet) -> Vec<Scored> {
        let Some(query) = self.vectors.get(node) else {
            return Vec::new();
        };
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let mut current = entry;
        let mut layer = self.max_level;
        while layer > 0 {
            current = self.greedy_descend(query, current, layer);
            layer -= 1;
        }
        let ef = top_k.saturating_add(1).max(HNSW_EF_SEARCH_MIN);
        let mut found = self.search_layer(query, &[current], ef, 0, visit);
        found.retain(|scored| scored.node as usize != node);
        found.truncate(top_k);
        found
    }

    fn candidate_pairs(&self, top_k: usize) -> CandidateSearchOutcome {
        let mut visit = VisitSet::new(self.vectors.len());
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut candidates = Vec::new();
        let mut raw_neighbors = 0usize;

        for node in 0..self.vectors.len() {
            let found = self.neighbors_of(node, top_k, &mut visit);
            raw_neighbors += found.len();
            for scored in found {
                let other = scored.node as usize;
                let (left, right) = if node < other {
                    (node, other)
                } else {
                    (other, node)
                };
                if !seen.insert((left, right)) {
                    continue;
                }
                candidates.push(AnnCandidate {
                    left,
                    right,
                    cosine: 1.0 - scored.distance,
                });
            }
        }

        let unique_pairs = candidates.len();
        CandidateSearchOutcome {
            mode: CandidateSearchMode::Ann,
            candidates,
            queries: self.vectors.len(),
            raw_neighbors,
            unique_pairs,
        }
    }
}

/// Deterministic level draw.
///
/// HNSW normally samples an exponential; deriving it from the node index keeps
/// the graph reproducible, which matters when a test asserts recall.
fn level_for(node: usize, level_scale: f64) -> usize {
    let mut bits = (node as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678_9ABC_DEF0);
    bits ^= bits >> 33;
    bits = bits.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    bits ^= bits >> 33;
    let unit = ((bits >> 11) as f64) / ((1u64 << 53) as f64);
    let unit = if unit <= 0.0 { f64::MIN_POSITIVE } else { unit };
    let level = (-unit.ln() * level_scale).floor();
    if level.is_finite() && level > 0.0 {
        (level as usize).min(HNSW_MAX_LEVEL)
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random unit vectors, so a failing assertion is
    /// reproducible rather than a coin flip.
    fn synthetic(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        };
        (0..count)
            .map(|_| {
                let raw: Vec<f32> = (0..dimension).map(|_| next()).collect();
                normalized(&raw)
            })
            .collect()
    }

    fn as_slices(vectors: &[Vec<f32>]) -> Vec<&[f32]> {
        vectors.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn every_pair_is_canonical_and_appears_once() {
        // Two vectors that are each other's nearest neighbour, so both queries
        // propose the same pair from opposite directions.
        let vectors = vec![
            normalized(&[1.0, 0.0, 0.0, 0.0]),
            normalized(&[0.99, 0.01, 0.0, 0.0]),
            normalized(&[0.0, 0.0, 0.0, 1.0]),
        ];
        let index = HnswIndex::build(&as_slices(&vectors)).unwrap();
        let outcome = index.candidate_pairs(8);

        let mut keys: Vec<(usize, usize)> = outcome
            .candidates
            .iter()
            .map(|candidate| (candidate.left, candidate.right))
            .collect();
        keys.sort();
        let mut deduped = keys.clone();
        deduped.dedup();
        assert_eq!(keys, deduped, "a pair was emitted more than once: {keys:?}");
        for candidate in &outcome.candidates {
            assert!(
                candidate.left < candidate.right,
                "pair ({}, {}) is not canonical",
                candidate.left,
                candidate.right
            );
        }
        assert_eq!(outcome.unique_pairs, outcome.candidates.len());
    }

    #[test]
    fn a_small_library_stays_on_brute_force() {
        let vectors = synthetic(100, 16, 7);
        let outcome = search_candidates(&as_slices(&vectors), ANN_TOP_K);
        assert_eq!(outcome.mode, CandidateSearchMode::BruteForce);
        // Brute force means every pair, which is the point of the threshold:
        // at this size the exhaustive answer is cheap and exact.
        assert_eq!(outcome.candidates.len(), 100 * 99 / 2);
        assert_eq!(outcome.queries, 0);
    }

    #[test]
    fn a_large_library_uses_the_graph_and_prunes_hard() {
        let count = 1_000;
        let vectors = synthetic(count, 24, 99);
        let outcome = search_candidates(&as_slices(&vectors), ANN_TOP_K);

        assert_eq!(outcome.mode, CandidateSearchMode::Ann);
        assert_eq!(outcome.queries, count);

        let exhaustive = count * (count - 1) / 2;
        assert!(
            outcome.candidates.len() < exhaustive / 4,
            "ANN produced {} candidates, barely below the exhaustive {exhaustive}",
            outcome.candidates.len()
        );
        // Each query can contribute at most top_k pairs, so the whole run is
        // bounded by n*k rather than n^2.
        assert!(
            outcome.candidates.len() <= count * ANN_TOP_K,
            "candidate count {} exceeded the n*k bound",
            outcome.candidates.len()
        );
        assert_eq!(outcome.unique_pairs, outcome.candidates.len());
        assert!(outcome.raw_neighbors >= outcome.unique_pairs);
    }

    #[test]
    fn a_very_large_library_stays_bounded_by_top_k() {
        let count = 5_000;
        let vectors = synthetic(count, 24, 123);
        let outcome = search_candidates(&as_slices(&vectors), ANN_TOP_K);

        assert_eq!(outcome.mode, CandidateSearchMode::Ann);
        assert_eq!(outcome.queries, count);
        assert!(
            outcome.candidates.len() <= count * ANN_TOP_K,
            "candidate count {} exceeded the n*k bound",
            outcome.candidates.len()
        );
        let exhaustive = count * (count - 1) / 2;
        assert!(
            outcome.candidates.len() < exhaustive / 20,
            "{} candidate pairs is not meaningfully below exhaustive {exhaustive}",
            outcome.candidates.len()
        );
    }

    #[test]
    fn an_obvious_near_neighbour_is_recalled() {
        // A haystack, plus one pair that differs by a whisker.
        let mut vectors = synthetic(ANN_MIN_LIBRARY_SIZE + 120, 24, 31);
        let base = vectors[0].clone();
        let nudged: Vec<f32> = base
            .iter()
            .enumerate()
            .map(|(index, value)| value + if index == 0 { 0.001 } else { 0.0 })
            .collect();
        let nudged_index = vectors.len();
        vectors.push(normalized(&nudged));

        let outcome = search_candidates(&as_slices(&vectors), ANN_TOP_K);
        assert_eq!(outcome.mode, CandidateSearchMode::Ann);

        let recalled = outcome
            .candidates
            .iter()
            .any(|candidate| candidate.left == 0 && candidate.right == nudged_index);
        assert!(
            recalled,
            "the near-identical vector was not proposed as a candidate"
        );
    }

    #[test]
    fn a_degenerate_input_does_not_panic() {
        assert!(HnswIndex::build(&[]).is_err());
        let empty: Vec<f32> = Vec::new();
        assert!(HnswIndex::build(&[empty.as_slice()]).is_err());

        let ragged_a = vec![1.0f32, 0.0];
        let ragged_b = vec![1.0f32, 0.0, 0.0];
        assert!(HnswIndex::build(&[ragged_a.as_slice(), ragged_b.as_slice()]).is_err());

        // A single vector, and none at all, are answered without a graph.
        let one = vec![1.0f32, 0.0];
        let outcome = search_candidates(&[one.as_slice()], ANN_TOP_K);
        assert!(outcome.candidates.is_empty());
        let outcome = search_candidates(&[], ANN_TOP_K);
        assert!(outcome.candidates.is_empty());
    }

    #[test]
    fn a_zero_vector_is_left_alone_rather_than_producing_nan() {
        let zero = vec![0.0f32; 8];
        let unit = normalized(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let outcome = search_candidates(&[zero.as_slice(), unit.as_slice()], ANN_TOP_K);
        for candidate in &outcome.candidates {
            assert!(
                candidate.cosine.is_finite(),
                "cosine was not finite: {}",
                candidate.cosine
            );
        }
    }

    #[test]
    fn descriptor_records_what_an_index_was_built_from() {
        let descriptor = AnnDescriptor::new("lib-1", Some("model-hash"), 384, "float16");
        assert_eq!(descriptor.index_version, ANN_INDEX_VERSION);
        assert_eq!(descriptor.dimension, 384);
        assert_ne!(
            descriptor,
            AnnDescriptor::new("lib-1", Some("other-hash"), 384, "float16")
        );
    }
}
