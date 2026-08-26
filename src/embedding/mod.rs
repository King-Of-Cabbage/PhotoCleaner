use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use half::f16;
use image::imageops::FilterType;
use ndarray::{s, Array4};
use ort::{
    CUDAExecutionProvider, EnvironmentGlobalThreadPoolOptions, GraphOptimizationLevel, Session,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::paths::PortablePaths;
use crate::scan_planner::{
    EMBEDDING_DIMENSION, EMBEDDING_DTYPE, EMBEDDING_MODEL_ID, EMBEDDING_PREPROCESS_VERSION,
};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

const INPUT_SIZE: u32 = 224;
const DEFAULT_CPU_BATCH: usize = 4;
const AI_QUEUE_LIMIT: usize = 32;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AiStatus {
    pub model_name: String,
    pub model_path: PathBuf,
    pub model_exists: bool,
    pub model_hash: Option<String>,
    pub runtime_path: PathBuf,
    pub runtime_exists: bool,
    pub runtime_loaded: bool,
    pub model_loaded: bool,
    pub cpu_available: bool,
    pub cuda_available: bool,
    pub device: String,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AiTestResult {
    pub success: bool,
    pub device: String,
    pub elapsed_ms: u128,
    pub output_dim: usize,
    pub has_nan_or_inf: bool,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuTopology {
    pub cpu_name: String,
    pub physical_cores: usize,
    pub logical_processors: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AiRuntimeProfile {
    pub device: String,
    pub cpu_threads: usize,
    pub batch_size: usize,
    pub execution_mode: String,
    pub graph_optimization: String,
    pub thread_spinning: String,
    pub nested_parallelism_prevented: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AiBenchmarkResult {
    pub topology: CpuTopology,
    pub profile: AiRuntimeProfile,
    pub images: usize,
    pub elapsed_ms: u128,
    pub images_per_second: f64,
    pub output_dim: usize,
    pub has_nan_or_inf: bool,
    pub success: bool,
    pub message: String,
}

pub struct AiInferenceEngine {
    tx: Option<Sender<InferenceRequest>>,
    handle: Option<thread::JoinHandle<()>>,
    profile: AiRuntimeProfile,
}

struct InferenceRequest {
    input: Array4<f32>,
    reply: Sender<Result<Vec<u8>, String>>,
}

pub fn environment_check(paths: &PortablePaths) -> AiStatus {
    let model_path = model_path(paths);
    let runtime_path = runtime_path(paths);
    let model_exists = model_path.exists();
    let runtime_exists = runtime_path.exists();
    let model_hash = model_exists.then(|| hash_file(&model_path)).flatten();

    let runtime_loaded = runtime_exists && init_runtime(paths).is_ok();
    let mut model_loaded = false;
    let mut cpu_available = false;
    let mut cuda_available = false;
    let detail;

    if !model_exists {
        detail = format!("AI模型缺失：{}", model_path.display());
    } else if !runtime_exists {
        detail = format!("ONNX Runtime加载失败：{}", runtime_path.display());
    } else if !runtime_loaded {
        detail = init_runtime(paths)
            .err()
            .unwrap_or_else(|| format!("ONNX Runtime加载失败：{}", runtime_path.display()));
    } else {
        match build_cpu_session(paths) {
            Ok(_) => {
                model_loaded = true;
                cpu_available = true;
                cuda_available = try_build_cuda_session(paths).is_ok();
                detail = if cuda_available {
                    "模型加载成功；CPU Execution Provider 可用；CUDA Execution Provider 可用"
                        .to_string()
                } else {
                    "模型加载成功；CPU Execution Provider 可用；CUDA不可用，已自动切换CPU"
                        .to_string()
                };
            }
            Err(err) => {
                detail = err.to_string();
            }
        }
    }

    let device = if cuda_available {
        "CUDA".to_string()
    } else if cpu_available {
        "CPU".to_string()
    } else {
        "不可用".to_string()
    };

    AiStatus {
        model_name: "DINOv2 ViT-S/14".to_string(),
        model_path,
        model_exists,
        model_hash,
        runtime_path,
        runtime_exists,
        runtime_loaded,
        model_loaded,
        cpu_available,
        cuda_available,
        device,
        detail,
    }
}

pub fn ensure_deep_available(paths: &PortablePaths) -> Result<String> {
    let status = environment_check(paths);
    if !status.model_exists {
        bail!("AI模型缺失：{}", status.model_path.display());
    }
    if !status.runtime_exists {
        bail!("ONNX Runtime加载失败：{}", status.runtime_path.display());
    }
    if !status.runtime_loaded {
        bail!("{}", status.detail);
    }
    if !status.model_loaded || !status.cpu_available {
        bail!("ONNX Runtime CPU模式无法运行：{}", status.detail);
    }
    Ok(status.model_hash.unwrap_or_default())
}

pub fn test_ai(paths: &PortablePaths) -> AiTestResult {
    let started = Instant::now();
    match run_generated_inference(paths) {
        Ok((device, embedding)) => AiTestResult {
            success: true,
            device,
            elapsed_ms: started.elapsed().as_millis(),
            output_dim: embedding.len(),
            has_nan_or_inf: embedding.iter().any(|v| !v.is_finite()),
            message: "模型加载成功".to_string(),
        },
        Err(err) => AiTestResult {
            success: false,
            device: "不可用".to_string(),
            elapsed_ms: started.elapsed().as_millis(),
            output_dim: 0,
            has_nan_or_inf: false,
            message: err.to_string(),
        },
    }
}

pub fn model_hash(paths: &PortablePaths) -> Option<String> {
    hash_file(&model_path(paths))
}

pub fn embed_image_file(paths: &PortablePaths, image_path: &Path) -> Result<Vec<u8>> {
    ensure_deep_available(paths)?;
    let input = preprocess_image(paths, image_path)?;
    let (_device, embedding) = run_with_fallback(paths, input)?;
    Ok(to_float16_blob(&embedding))
}

impl AiInferenceEngine {
    pub fn start(paths: PortablePaths) -> Result<Self> {
        ensure_deep_available(&paths)?;
        let profile = runtime_profile("CPU");
        let (tx, rx) = bounded::<InferenceRequest>(AI_QUEUE_LIMIT);
        let thread_profile = profile.clone();
        let handle = thread::spawn(move || ai_inference_loop(paths, rx, thread_profile));
        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            profile,
        })
    }

    pub fn embed_image_file(&self, paths: &PortablePaths, image_path: &Path) -> Result<Vec<u8>> {
        let input = preprocess_image(paths, image_path)?;
        self.embed_preprocessed(input)
    }

    pub fn embed_preprocessed(&self, input: Array4<f32>) -> Result<Vec<u8>> {
        let (reply, result) = bounded(1);
        let tx = self
            .tx
            .as_ref()
            .context("AI inference engine is already stopped")?;
        tx.send(InferenceRequest { input, reply })?;
        result
            .recv()
            .context("AI inference engine stopped before returning a result")?
            .map_err(anyhow::Error::msg)
    }

    pub fn profile(&self) -> &AiRuntimeProfile {
        &self.profile
    }
}

impl Drop for AiInferenceEngine {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_generated_inference(paths: &PortablePaths) -> Result<(String, Vec<f32>)> {
    ensure_deep_available(paths)?;
    let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for y in 0..INPUT_SIZE as usize {
        for x in 0..INPUT_SIZE as usize {
            input[[0, 0, y, x]] = ((x % 255) as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            input[[0, 1, y, x]] = ((y % 255) as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            input[[0, 2, y, x]] =
                (((x + y) % 255) as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }
    }
    run_with_fallback(paths, input)
}

fn run_with_fallback(paths: &PortablePaths, input: Array4<f32>) -> Result<(String, Vec<f32>)> {
    if cuda_runtime_present(paths) {
        match run_in_session(try_build_cuda_session(paths)?, &input) {
            Ok(values) => return Ok(("CUDA".to_string(), values)),
            Err(cuda_err) => {
                eprintln!("CUDA初始化失败，已自动切换CPU：{cuda_err}");
            }
        }
    }
    let values = run_in_session(build_cpu_session(paths)?, &input)?;
    Ok(("CPU".to_string(), values))
}

fn run_in_session(mut session: Session, input: &Array4<f32>) -> Result<Vec<f32>> {
    run_in_session_ref(&mut session, input)
}

fn run_in_session_ref(session: &mut Session, input: &Array4<f32>) -> Result<Vec<f32>> {
    let outputs = session.run(ort::inputs![input.clone()]?)?;
    let first = outputs.values().next().context("模型没有输出")?;
    let data = first.try_extract_tensor::<f32>()?;
    let mut values: Vec<f32> = data.iter().copied().collect();
    if values.len() != EMBEDDING_DIMENSION as usize {
        if values.len() > EMBEDDING_DIMENSION as usize {
            values.truncate(EMBEDDING_DIMENSION as usize);
        } else {
            bail!("输出维度不是384：{}", values.len());
        }
    }
    l2_normalize(&mut values);
    Ok(values)
}

fn run_batch_in_session(session: &mut Session, inputs: &[Array4<f32>]) -> Result<Vec<Vec<u8>>> {
    let batch_len = inputs.len();
    if batch_len == 0 {
        return Ok(Vec::new());
    }
    let mut batch = Array4::<f32>::zeros((batch_len, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for (idx, input) in inputs.iter().enumerate() {
        batch
            .slice_mut(s![idx..idx + 1, .., .., ..])
            .assign(&input.view());
    }
    let outputs = session.run(ort::inputs![batch]?)?;
    let first = outputs
        .values()
        .next()
        .context("model did not return output")?;
    let data = first.try_extract_tensor::<f32>()?;
    let values: Vec<f32> = data.iter().copied().collect();
    let dim = EMBEDDING_DIMENSION as usize;
    if values.len() != batch_len * dim {
        bail!(
            "Output dimension is not batch*384: {} images -> {} values",
            batch_len,
            values.len()
        );
    }
    Ok(values
        .chunks_exact(dim)
        .map(|chunk| {
            let mut embedding = chunk.to_vec();
            l2_normalize(&mut embedding);
            to_float16_blob(&embedding)
        })
        .collect())
}

fn ai_inference_loop(
    paths: PortablePaths,
    rx: Receiver<InferenceRequest>,
    profile: AiRuntimeProfile,
) {
    let mut session = match build_cpu_session_with_threads(&paths, profile.cpu_threads) {
        Ok(session) => session,
        Err(err) => {
            while let Ok(request) = rx.recv() {
                let _ = request.reply.send(Err(err.to_string()));
            }
            return;
        }
    };
    while let Ok(first) = rx.recv() {
        let mut requests = vec![first];
        let deadline = Instant::now() + Duration::from_millis(5);
        while requests.len() < profile.batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok(request) => requests.push(request),
                Err(_) => break,
            }
        }
        let inputs: Vec<_> = requests
            .iter()
            .map(|request| request.input.clone())
            .collect();
        match run_batch_in_session(&mut session, &inputs) {
            Ok(embeddings) => {
                for (request, embedding) in requests.into_iter().zip(embeddings) {
                    let _ = request.reply.send(Ok(embedding));
                }
            }
            Err(err) => {
                if requests.len() > 1 {
                    for (request, input) in requests.into_iter().zip(inputs) {
                        let result = run_in_session_ref(&mut session, &input)
                            .map(|embedding| to_float16_blob(&embedding));
                        let _ = request.reply.send(result.map_err(|single_err| {
                            format!("{err}; single fallback failed: {single_err}")
                        }));
                    }
                } else {
                    let message = err.to_string();
                    for request in requests {
                        let _ = request.reply.send(Err(message.clone()));
                    }
                }
            }
        }
    }
}

fn build_cpu_session(paths: &PortablePaths) -> Result<Session> {
    build_cpu_session_with_threads(paths, ai_cpu_threads())
}

fn build_cpu_session_with_threads(paths: &PortablePaths, threads: usize) -> Result<Session> {
    init_runtime(paths).map_err(anyhow::Error::msg)?;
    Session::builder()?
        .with_intra_threads(threads)?
        .with_inter_threads(1)?
        .with_parallel_execution(false)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path(paths))
        .map_err(|err| anyhow::anyhow!("模型加载失败：{}；{err:?}", model_path(paths).display()))
}

fn try_build_cuda_session(paths: &PortablePaths) -> Result<Session> {
    init_runtime(paths).map_err(anyhow::Error::msg)?;
    if !cuda_runtime_present(paths) {
        bail!("CUDA组件不存在");
    }
    Session::builder()?
        .with_parallel_execution(false)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(model_path(paths))
        .context("CUDA Execution Provider 初始化失败")
}

fn init_runtime(paths: &PortablePaths) -> std::result::Result<(), String> {
    ORT_INIT
        .get_or_init(|| {
            let dll = runtime_path(paths);
            if !dll.exists() {
                return Err(format!("ONNX Runtime加载失败：{}", dll.display()));
            }
            if let Some(dir) = dll.parent() {
                let old_path = std::env::var("PATH").unwrap_or_default();
                let dir_text = dir.display().to_string();
                if !old_path
                    .split(';')
                    .any(|entry| entry.eq_ignore_ascii_case(&dir_text))
                {
                    std::env::set_var("PATH", format!("{dir_text};{old_path}"));
                }
            }
            ort::init_from(dll.to_string_lossy())
                .with_global_thread_pool(EnvironmentGlobalThreadPoolOptions {
                    intra_op_parallelism: Some(ai_cpu_threads() as i32),
                    inter_op_parallelism: Some(1),
                    spin_control: Some(false),
                    intra_op_thread_affinity: None,
                })
                .commit()
                .map(|_| ())
                .map_err(|err| format!("ONNX Runtime加载失败：{}；{}", dll.display(), err))
        })
        .clone()
}

fn preprocess_image(paths: &PortablePaths, path: &Path) -> Result<Array4<f32>> {
    let rgb = decode_image_rgb(paths, path, INPUT_SIZE)?;
    if rgb.len() != (INPUT_SIZE * INPUT_SIZE * 3) as usize {
        bail!(
            "AI图像解码失败：{}；RGB输出尺寸不正确：{} bytes",
            path.display(),
            rgb.len()
        );
    }
    let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for y in 0..INPUT_SIZE as usize {
        for x in 0..INPUT_SIZE as usize {
            let offset = (y * INPUT_SIZE as usize + x) * 3;
            for channel in 0..3 {
                let value = rgb[offset + channel] as f32 / 255.0;
                input[[0, channel, y, x]] =
                    (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
            }
        }
    }
    Ok(input)
}

pub fn decode_image_rgb(paths: &PortablePaths, path: &Path, size: u32) -> Result<Vec<u8>> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let rgb = if matches!(extension.as_str(), "heic" | "heif") {
        decode_heif_with_ffmpeg(paths, path, size)?
    } else {
        image::open(path)
            .with_context(|| format!("AI图像解码失败：{}", path.display()))?
            .resize_exact(size, size, FilterType::Triangle)
            .to_rgb8()
            .into_raw()
    };
    Ok(rgb)
}

pub fn decoded_image_dimensions(paths: &PortablePaths, path: &Path) -> Result<(u32, u32)> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if matches!(extension.as_str(), "heic" | "heif") {
        let png = decode_heif_png_bytes(paths, path)?;
        let image = image::load_from_memory(&png)
            .with_context(|| format!("HEIC decoded dimensions failed: {}", path.display()))?;
        return Ok((image.width(), image.height()));
    }
    image::image_dimensions(path)
        .with_context(|| format!("image dimensions failed: {}", path.display()))
}

pub fn cpu_topology() -> CpuTopology {
    CpuTopology {
        cpu_name: cpu_name(),
        physical_cores: num_cpus::get_physical().max(1),
        logical_processors: num_cpus::get().max(1),
    }
}

pub fn runtime_profile(device: &str) -> AiRuntimeProfile {
    AiRuntimeProfile {
        device: device.to_string(),
        cpu_threads: ai_cpu_threads(),
        batch_size: DEFAULT_CPU_BATCH,
        execution_mode: "ORT_SEQUENTIAL".to_string(),
        graph_optimization: "ORT_ENABLE_ALL".to_string(),
        thread_spinning: "OFF".to_string(),
        nested_parallelism_prevented: true,
    }
}

pub fn benchmark_generated(paths: &PortablePaths, images: usize) -> AiBenchmarkResult {
    let topology = cpu_topology();
    let profile = runtime_profile("CPU");
    let started = Instant::now();
    let mut has_nan_or_inf = false;
    let mut output_dim = 0;
    let result = (|| -> Result<()> {
        ensure_deep_available(paths)?;
        let engine = AiInferenceEngine::start(paths.clone())?;
        let mut outputs = Vec::with_capacity(images);
        for idx in 0..images {
            outputs.push(engine.embed_preprocessed(generated_input(idx))?);
        }
        output_dim = EMBEDDING_DIMENSION as usize;
        has_nan_or_inf = outputs.iter().any(|bytes| {
            bytes.chunks_exact(2).any(|chunk| {
                let value = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                !value.is_finite()
            })
        });
        Ok(())
    })();
    let elapsed_ms = started.elapsed().as_millis();
    let images_per_second = if elapsed_ms == 0 {
        0.0
    } else {
        images as f64 / (elapsed_ms as f64 / 1000.0)
    };
    AiBenchmarkResult {
        topology,
        profile,
        images,
        elapsed_ms,
        images_per_second,
        output_dim,
        has_nan_or_inf,
        success: result.is_ok(),
        message: result
            .map(|_| "benchmark completed".to_string())
            .unwrap_or_else(|err| err.to_string()),
    }
}

fn generated_input(seed: usize) -> Array4<f32> {
    let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for y in 0..INPUT_SIZE as usize {
        for x in 0..INPUT_SIZE as usize {
            input[[0, 0, y, x]] =
                (((x + seed) % 255) as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            input[[0, 1, y, x]] =
                (((y + seed * 3) % 255) as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            input[[0, 2, y, x]] =
                (((x + y + seed * 7) % 255) as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }
    }
    input
}

fn ai_cpu_threads() -> usize {
    num_cpus::get_physical().max(1)
}

fn cpu_name() -> String {
    std::env::var("PROCESSOR_IDENTIFIER")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "Unknown CPU".to_string())
}

fn decode_heif_with_ffmpeg(paths: &PortablePaths, path: &Path, size: u32) -> Result<Vec<u8>> {
    let png = decode_heif_png_bytes(paths, path)?;
    return Ok(image::load_from_memory(&png)
        .with_context(|| format!("HEIC decoded output failed: {}", path.display()))?
        .resize_exact(size, size, FilterType::Triangle)
        .to_rgb8()
        .into_raw());
}

fn decode_heif_png_bytes(paths: &PortablePaths, path: &Path) -> Result<Vec<u8>> {
    let ffmpeg = paths.runtime_media_dir.join("ffmpeg.exe");
    if !ffmpeg.exists() {
        bail!("HEIC解码器缺失：{}", ffmpeg.display());
    }
    let input_path = ffmpeg_input_path(path);
    let output = Command::new(&ffmpeg)
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(input_path)
        .arg("-frames:v")
        .arg("1")
        .arg("-f")
        .arg("image2pipe")
        .arg("-vcodec")
        .arg("png")
        .arg("pipe:1")
        .output()
        .with_context(|| format!("HEIC解码器启动失败：{}", ffmpeg.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("HEIC解码失败：{}；{}", path.display(), stderr.trim());
    }
    return Ok(output.stdout);
    /*
    Ok(image::load_from_memory(&output.stdout)
        .with_context(|| format!("HEIC解码输出无法读取：{}", path.display()))?
        .resize_exact(size, size, FilterType::Triangle)
        .to_rgb8()
        .into_raw())
    */
}

fn ffmpeg_input_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    text.strip_prefix(r"\\?\").unwrap_or(&text).to_string()
}

fn to_float16_blob(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&f16::from_f32(*value).to_le_bytes());
    }
    bytes
}

fn hash_file(path: &Path) -> Option<String> {
    fs::read(path)
        .ok()
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
}

fn model_path(paths: &PortablePaths) -> PathBuf {
    paths.models_dir.join("dinov2_vits14.onnx")
}

fn runtime_path(paths: &PortablePaths) -> PathBuf {
    paths.runtime_onnx_dir.join("onnxruntime.dll")
}

fn cuda_runtime_present(paths: &PortablePaths) -> bool {
    paths
        .runtime_onnx_dir
        .join("onnxruntime_providers_cuda.dll")
        .exists()
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for value in values {
            *value = (*value as f64 / norm) as f32;
        }
    }
}

pub fn embedding_cache_metadata(
    paths: &PortablePaths,
) -> (Option<String>, i64, &'static str, i64, &'static str) {
    (
        model_hash(paths),
        EMBEDDING_PREPROCESS_VERSION,
        EMBEDDING_DTYPE,
        EMBEDDING_DIMENSION,
        EMBEDDING_MODEL_ID,
    )
}
