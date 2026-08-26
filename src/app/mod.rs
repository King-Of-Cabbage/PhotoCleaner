use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use eframe::egui;

use crate::config::Settings;
use crate::database::{
    CleanupAsset, CleanupGroup, CleanupResults, Database, Library, MediaCounts, MoveOperation,
};
use crate::embedding::{AiStatus, AiTestResult};
use crate::paths::{PortablePaths, PORTABLE_WRITE_ERROR};
use crate::scanner::{ScanMode, ScanOutcome, ScanProgress, ScanStage};
use crate::tasks::{self, TaskCommand, TaskEvent, TaskRunner};
use crate::thumbnails::{LruBudget, ThumbnailService};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Page {
    Home,
    Results,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResultsTab {
    Duplicates,
    Near,
    Burst,
    Similar,
    Pending,
}

impl ResultsTab {
    fn label(self) -> &'static str {
        match self {
            Self::Duplicates => "完全重复",
            Self::Near => "近重复",
            Self::Burst => "连拍",
            Self::Similar => "相似照片",
            Self::Pending => "待删除",
        }
    }
}

struct ResultsState {
    tab: ResultsTab,
    selected_files: HashSet<i64>,
    pending_assets: BTreeMap<i64, CleanupAsset>,
    ignored_groups: HashSet<String>,
    compare_group: Option<(String, i64)>,
    compare_file_id: Option<i64>,
    large_asset: Option<CleanupAsset>,
    message: String,
    page: usize,
    thumbnail_errors: HashMap<String, String>,
}

impl Default for ResultsState {
    fn default() -> Self {
        Self {
            tab: ResultsTab::Duplicates,
            selected_files: HashSet::new(),
            pending_assets: BTreeMap::new(),
            ignored_groups: HashSet::new(),
            compare_group: None,
            compare_file_id: None,
            large_asset: None,
            message: String::new(),
            page: 0,
            thumbnail_errors: HashMap::new(),
        }
    }
}

pub struct PhotoCleanerApp {
    paths: PortablePaths,
    settings: Settings,
    task_runner: TaskRunner,
    libraries: Vec<Library>,
    selected_library: Option<PathBuf>,
    scan_mode: ScanMode,
    status: String,
    progress: Option<ScanProgress>,
    last_outcome: Option<ScanOutcome>,
    media_counts: MediaCounts,
    is_scanning: bool,
    portable_error: Option<String>,
    ai_status: AiStatus,
    ai_test_result: Option<AiTestResult>,
    page: Page,
    cleanup_results: CleanupResults,
    results: ResultsState,
    texture_cache: HashMap<String, egui::TextureHandle>,
    texture_budget: LruBudget<String>,
    thumbnails: ThumbnailService,
    confirm_stage_pending: bool,
}

impl PhotoCleanerApp {
    pub fn new(paths: PortablePaths) -> Self {
        let portable_error = paths
            .assert_writable()
            .err()
            .map(|_| PORTABLE_WRITE_ERROR.to_string());
        let settings = Settings::load_or_create(&paths).unwrap_or_default();
        let thumbnail_budget_bytes = settings
            .thumbnail_cache_limit_mb
            .saturating_mul(1024 * 1024);
        let thumbnails = ThumbnailService::start(&paths, thumbnail_budget_bytes);
        let (libraries, media_counts) = tasks::refresh_counts(&paths);
        let ai_status = crate::embedding::environment_check(&paths);
        let cleanup_results = Database::open(&paths)
            .and_then(|db| db.load_cleanup_results())
            .unwrap_or_default();
        let mut app = Self {
            task_runner: TaskRunner::start(paths.clone()),
            paths,
            settings,
            libraries,
            selected_library: None,
            scan_mode: ScanMode::Standard,
            status: "就绪".to_string(),
            progress: None,
            last_outcome: None,
            media_counts,
            is_scanning: false,
            portable_error,
            ai_status,
            ai_test_result: None,
            page: Page::Home,
            cleanup_results,
            results: ResultsState::default(),
            texture_cache: HashMap::new(),
            texture_budget: LruBudget::new(thumbnail_budget_bytes),
            thumbnails,
            confirm_stage_pending: false,
        };
        app.preselect_duplicate_candidates();
        app
    }

    fn handle_events(&mut self) {
        for event in self.task_runner.drain_events() {
            match event {
                TaskEvent::ScanStarted { root, mode } => {
                    self.is_scanning = true;
                    self.last_outcome = None;
                    self.status = format!("正在{}：{}", mode.label(), root.display());
                }
                TaskEvent::ScanProgress(progress) => {
                    self.status = progress.activity.clone();
                    self.progress = Some(progress);
                }
                TaskEvent::ScanFinished { outcome } => {
                    self.is_scanning = false;
                    self.status = "扫描完成".to_string();
                    self.last_outcome = Some(outcome);
                    let (libraries, media_counts) = tasks::refresh_counts(&self.paths);
                    self.libraries = libraries;
                    self.media_counts = media_counts;
                    self.refresh_cleanup_results();
                }
                TaskEvent::Failed(message) => {
                    self.is_scanning = false;
                    self.status = format!("任务失败：{message}");
                }
            }
        }
    }

    fn refresh_cleanup_results(&mut self) {
        match Database::open(&self.paths).and_then(|db| db.load_cleanup_results()) {
            Ok(results) => {
                self.cleanup_results = results;
                self.preselect_duplicate_candidates();
            }
            Err(err) => self.results.message = format!("读取清理结果失败：{err}"),
        }
    }

    /// Only byte-identical copies are pre-selected for deletion.
    ///
    /// NEAR_DUPLICATE, BURST_SIMILAR and VISUALLY_SIMILAR are similarity
    /// judgements, not proof that two files hold the same picture, so nothing
    /// in those groups is ever ticked on the user's behalf.
    fn preselect_duplicate_candidates(&mut self) {
        self.results.selected_files.clear();
        for group in &self.cleanup_results.duplicate_groups {
            if group.kind != EXACT_DUPLICATE {
                continue;
            }
            for member in &group.members {
                if !member.is_recommended_keep {
                    self.results.selected_files.insert(member.file_id);
                }
            }
        }
    }
}

impl eframe::App for PhotoCleanerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_events();
        self.drain_thumbnails(ctx);
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.results.large_asset = None;
            self.results.compare_group = None;
        }
        ctx.request_repaint_after(Duration::from_millis(100));

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("PhotoCleaner");
                ui.separator();
                ui.label(&self.status);
                ui.separator();
                if ui
                    .selectable_label(self.page == Page::Home, "扫描")
                    .clicked()
                {
                    self.page = Page::Home;
                }
                if ui
                    .selectable_label(self.page == Page::Results, "清理结果")
                    .clicked()
                {
                    self.page = Page::Results;
                    self.refresh_cleanup_results();
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.page {
            Page::Home => self.draw_home(ui),
            Page::Results => self.draw_results_page(ctx, ui),
        });

        self.draw_large_image_window(ctx);
        self.draw_compare_window(ctx);
    }
}

impl PhotoCleanerApp {
    fn draw_home(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if let Some(error) = &self.portable_error {
                    ui.colored_label(egui::Color32::from_rgb(180, 40, 40), error);
                    ui.separator();
                }

                ui.horizontal(|ui| {
                    if ui.button("选择照片文件夹").clicked() {
                        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                            self.selected_library = Some(folder);
                        }
                    }
                    let selected = self
                        .selected_library
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "尚未选择".to_string());
                    ui.label(selected);
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.scan_mode, ScanMode::Standard, "标准扫描");
                    ui.radio_value(&mut self.scan_mode, ScanMode::Deep, "深度扫描");
                    if ui
                        .add_enabled(
                            !self.is_scanning && self.selected_library.is_some(),
                            egui::Button::new("开始扫描"),
                        )
                        .clicked()
                    {
                        if let Some(root) = self.selected_library.clone() {
                            let _ = self.task_runner.sender().send(TaskCommand::Scan {
                                root,
                                mode: self.scan_mode,
                            });
                        }
                    }
                    ui.add_enabled(false, egui::Button::new("暂停"));
                    ui.add_enabled(false, egui::Button::new("取消"));
                });

                ui.separator();
                self.draw_progress(ui);

                ui.separator();
                self.draw_ai_status(ui);

                ui.separator();
                ui.heading("媒体库");
                ui.horizontal(|ui| {
                    ui.label(format!("资产：{}", self.media_counts.media_assets));
                    ui.label(format!("图片：{}", self.media_counts.images));
                    ui.label(format!("Live Photo：{}", self.media_counts.live_photos));
                    ui.label(format!("视频：{}", self.media_counts.videos));
                });

                ui.separator();
                ui.heading("历史扫描");
                if self.libraries.is_empty() {
                    ui.label("暂无照片库");
                } else {
                    egui::Grid::new("libraries").striped(true).show(ui, |ui| {
                        ui.label("照片库");
                        ui.label("路径");
                        ui.end_row();
                        for library in &self.libraries {
                            ui.label(&library.display_name);
                            ui.label(&library.last_known_root);
                            ui.end_row();
                        }
                    });
                }

                if let Some(outcome) = &self.last_outcome {
                    ui.separator();
                    ui.heading("扫描完成");
                    ui.label(format!("总耗时：{}", format_ms(outcome.summary.elapsed_ms)));
                    ui.label(format!("媒体：{}", outcome.summary.completed));
                    ui.label(format!("图片：{}", outcome.summary.images));
                    ui.label(format!("Live Photo：{}", outcome.summary.live_photos));
                    ui.label(format!("视频：{}", outcome.summary.videos));
                    ui.label(format!("新增：{}", outcome.summary.new_files));
                    ui.label(format!("更新：{}", outcome.summary.updated_files));
                    ui.label(format!("复用：{}", outcome.summary.reused_files));
                    ui.label(format!("AI分析新增：{}", outcome.summary.ai_computed));
                    ui.label(format!("无法解析：{}", outcome.summary.failed_files));
                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "完全重复 {}组",
                            self.tab_count(ResultsTab::Duplicates)
                        ));
                        if ui.button("查看##dup_done").clicked() {
                            self.results.tab = ResultsTab::Duplicates;
                            self.results.page = 0;
                            self.page = Page::Results;
                        }
                    });
                    for (tab, id) in [
                        (ResultsTab::Near, "near_done"),
                        (ResultsTab::Burst, "burst_done"),
                        (ResultsTab::Similar, "sim_done"),
                    ] {
                        let count = self.tab_count(tab);
                        ui.horizontal(|ui| {
                            ui.label(format!("{} {}组", tab.label(), count));
                            if ui.button(format!("查看##{id}")).clicked() {
                                self.results.tab = tab;
                                self.results.page = 0;
                                self.page = Page::Results;
                            }
                        });
                    }
                }
            });
    }

    fn draw_results_page(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for tab in [
                ResultsTab::Duplicates,
                ResultsTab::Near,
                ResultsTab::Burst,
                ResultsTab::Similar,
                ResultsTab::Pending,
            ] {
                let count = self.tab_count(tab);
                let label = format!("{} ({})", tab.label(), count);
                if ui
                    .selectable_label(self.results.tab == tab, label)
                    .clicked()
                {
                    self.results.tab = tab;
                    self.results.page = 0;
                }
            }
            ui.separator();
            if ui.button("刷新").clicked() {
                self.refresh_cleanup_results();
            }
        });

        if !self.results.message.is_empty() {
            ui.colored_label(egui::Color32::from_rgb(40, 100, 170), &self.results.message);
        }
        ui.separator();

        if self.results.tab == ResultsTab::Pending {
            self.draw_pending_tab(ui);
            return;
        }

        let visible: Vec<CleanupGroup> = self
            .groups_for_active_tab()
            .into_iter()
            .filter(|group| !self.results.ignored_groups.contains(&group_key(group)))
            .collect();
        if visible.is_empty() {
            ui.label("暂无可显示的组。");
            return;
        }

        // Only one page of groups is built per frame. egui walks the whole UI
        // tree every frame, so without this the first frame on a large library
        // would ask for a thumbnail for every photo in every group at once.
        let total_pages = visible.len().div_ceil(GROUPS_PER_PAGE).max(1);
        if self.results.page >= total_pages {
            self.results.page = total_pages - 1;
        }
        let page = self.results.page;
        let start = page * GROUPS_PER_PAGE;
        let end = (start + GROUPS_PER_PAGE).min(visible.len());

        ui.horizontal(|ui| {
            if ui
                .add_enabled(page > 0, egui::Button::new("上一页"))
                .clicked()
            {
                self.results.page = page.saturating_sub(1);
            }
            ui.label(format!(
                "第 {} / {} 页，共 {} 组（本页 {}–{}）",
                page + 1,
                total_pages,
                visible.len(),
                start + 1,
                end
            ));
            if ui
                .add_enabled(page + 1 < total_pages, egui::Button::new("下一页"))
                .clicked()
            {
                self.results.page = page + 1;
            }
            let pending = self.thumbnails.pending_len();
            if pending > 0 {
                ui.separator();
                ui.label(format!("缩略图载入中：{pending}"));
            }
        });
        ui.separator();

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for group in visible[start..end].to_vec() {
                    self.draw_group(ctx, ui, group);
                    ui.separator();
                }
            });
    }

    fn tab_count(&self, tab: ResultsTab) -> usize {
        match tab {
            ResultsTab::Pending => self.results.pending_assets.len(),
            other => self.groups_for_tab(other).len(),
        }
    }

    /// Each tab shows exactly one classification. They used to overlap: 近重复
    /// was hardcoded empty, and 相似照片 showed every similarity group
    /// including bursts.
    fn groups_for_tab(&self, tab: ResultsTab) -> Vec<CleanupGroup> {
        let (source, kind) = match tab {
            ResultsTab::Duplicates => (&self.cleanup_results.duplicate_groups, EXACT_DUPLICATE),
            ResultsTab::Near => (&self.cleanup_results.similarity_groups, NEAR_DUPLICATE),
            ResultsTab::Burst => (&self.cleanup_results.similarity_groups, BURST_SIMILAR),
            ResultsTab::Similar => (&self.cleanup_results.similarity_groups, VISUALLY_SIMILAR),
            ResultsTab::Pending => return Vec::new(),
        };
        source
            .iter()
            .filter(|group| group.kind == kind)
            .cloned()
            .collect()
    }

    fn groups_for_active_tab(&self) -> Vec<CleanupGroup> {
        self.groups_for_tab(self.results.tab)
    }

    fn draw_group(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, group: CleanupGroup) {
        let selected_count = group
            .members
            .iter()
            .filter(|member| self.results.selected_files.contains(&member.file_id))
            .count();
        let selected_bytes: u64 = group
            .members
            .iter()
            .filter(|member| self.results.selected_files.contains(&member.file_id))
            .map(|member| member.file_size)
            .sum();

        ui.horizontal(|ui| {
            ui.strong(format!("{} #{}", kind_label(&group.kind), group.id));
            ui.label(format!("{}项", group.members.len()));
            ui.label(format!("可释放约 {}", format_bytes(group.reclaim_bytes)));
            ui.label(format!("时间 {}", short_time(&group.created_at)));
        });

        ui.horizontal(|ui| {
            ui.label(evidence_text(&group));
            if ui.button("查看比较").clicked() {
                self.results.compare_group = Some((group.table_name.clone(), group.id));
                self.results.compare_file_id = group
                    .members
                    .iter()
                    .find(|member| !member.is_recommended_keep)
                    .or_else(|| group.members.first())
                    .map(|member| member.file_id);
            }
            if ui.button("选择清理").clicked() {
                for member in &group.members {
                    if !member.is_recommended_keep {
                        self.results.selected_files.insert(member.file_id);
                    }
                }
            }
            if group.kind == EXACT_DUPLICATE && ui.button("保留最佳，其余加入待删除").clicked()
            {
                for member in &group.members {
                    if !member.is_recommended_keep {
                        self.results
                            .pending_assets
                            .insert(member.asset_id, member.clone());
                    }
                }
                self.results.message = "已加入待删除，尚未移动文件。".to_string();
            }
            if ui.button("忽略此组").clicked() {
                self.results.ignored_groups.insert(group_key(&group));
            }
        });

        egui::ScrollArea::horizontal()
            .id_source(format!("group_scroll_{}_{}", group.table_name, group.id))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for member in &group.members {
                        self.draw_asset_card(ctx, ui, member);
                    }
                });
            });

        ui.horizontal(|ui| {
            ui.label(format!(
                "已选 {} 项，预计释放 {}",
                selected_count,
                format_bytes(selected_bytes)
            ));
            if ui.button("加入待删除").clicked() {
                for member in &group.members {
                    if self.results.selected_files.contains(&member.file_id) {
                        self.results
                            .pending_assets
                            .insert(member.asset_id, member.clone());
                    }
                }
                self.results.message = "已加入待删除，尚未移动文件。".to_string();
            }
        });
    }

    fn draw_asset_card(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, asset: &CleanupAsset) {
        ui.vertical(|ui| {
            ui.set_width(180.0);
            let mut selected = self.results.selected_files.contains(&asset.file_id);
            if ui.checkbox(&mut selected, "").changed() {
                if selected {
                    self.results.selected_files.insert(asset.file_id);
                } else {
                    self.results.selected_files.remove(&asset.file_id);
                }
            }
            match self.asset_texture(ctx, asset, 128) {
                Some(texture) => {
                    let image = egui::Image::new((texture.id(), egui::vec2(128.0, 128.0)));
                    if ui.add(egui::ImageButton::new(image)).clicked() {
                        self.results.large_asset = Some(asset.clone());
                    }
                }
                None => {
                    let key = texture_key(asset, 128);
                    let failed = self.results.thumbnail_errors.contains_key(&key);
                    if draw_thumbnail_placeholder(ui, 128.0, failed).clicked() {
                        self.results.large_asset = Some(asset.clone());
                    }
                }
            }
            if asset.is_recommended_keep {
                ui.colored_label(egui::Color32::from_rgb(20, 120, 70), "建议保留");
            }
            ui.label(&asset.file_name);
            ui.label(format!(
                "{} / {}",
                asset.asset_type,
                format_resolution(asset.width, asset.height)
            ));
            ui.label(format_bytes(asset.file_size));
            if let Some(time) = &asset.capture_time {
                ui.label(short_time(time));
            }
            if let Some(similarity) = asset.similarity {
                ui.label(format!("AI特征: {:.3}", similarity));
            }
            if let Some(distance) = asset.distance {
                ui.label(format!("pHash距离: {}", distance));
            }
            ui.horizontal(|ui| {
                if ui.button("大图").clicked() {
                    self.results.large_asset = Some(asset.clone());
                }
                if ui.button("路径").clicked() {
                    ui.output_mut(|output| {
                        output.copied_text = asset_path(asset).display().to_string()
                    });
                }
                if ui.button("文件夹").clicked() {
                    open_in_explorer(&asset_path(asset));
                }
            });
            if ui.button("保留此项").clicked() {
                self.results.selected_files.remove(&asset.file_id);
                self.results.pending_assets.remove(&asset.asset_id);
            }
        });
    }

    fn draw_pending_tab(&mut self, ui: &mut egui::Ui) {
        let pending: Vec<_> = self.results.pending_assets.values().cloned().collect();
        let total_bytes: u64 = pending.iter().map(|asset| asset.file_size).sum();
        ui.horizontal(|ui| {
            ui.strong(format!(
                "待删除 {} 个资产，预计释放 {}",
                pending.len(),
                format_bytes(total_bytes)
            ));
            if ui.button("撤销上次移动").clicked() {
                self.undo_latest_move();
            }
        });
        ui.horizontal(|ui| {
            if ui.button("移动到待删除文件夹").clicked() {
                self.confirm_stage_pending = true;
            }
            if self.confirm_stage_pending {
                ui.colored_label(
                    egui::Color32::from_rgb(170, 90, 20),
                    "再次点击确认移动，不会永久删除。",
                );
                if ui.button("确认移动").clicked() {
                    self.stage_pending_assets();
                    self.confirm_stage_pending = false;
                }
                if ui.button("取消").clicked() {
                    self.confirm_stage_pending = false;
                }
            }
        });
        ui.separator();

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for asset in pending {
                    ui.horizontal(|ui| {
                        ui.label(&asset.file_name);
                        ui.label(asset.asset_type.clone());
                        ui.label(format_bytes(asset.file_size));
                        if ui.button("还原到列表").clicked() {
                            self.results.pending_assets.remove(&asset.asset_id);
                        }
                        if ui.button("打开文件夹").clicked() {
                            open_in_explorer(&asset_path(&asset));
                        }
                    });
                }
            });
    }

    fn draw_large_image_window(&mut self, ctx: &egui::Context) {
        let Some(asset) = self.results.large_asset.clone() else {
            return;
        };
        let mut open = true;
        egui::Window::new("大图查看")
            .open(&mut open)
            .resizable(true)
            .vscroll(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.strong(&asset.file_name);
                    ui.label(format_resolution(asset.width, asset.height));
                    if ui.button("打开文件夹").clicked() {
                        open_in_explorer(&asset_path(&asset));
                    }
                });
                if let Some(texture) = self.asset_texture(ctx, &asset, 768) {
                    let available = ui.available_width().min(768.0);
                    ui.image((texture.id(), egui::vec2(available, available)));
                } else {
                    let key = texture_key(&asset, 768);
                    match self.results.thumbnail_errors.get(&key) {
                        Some(err) => {
                            ui.label(format!("无法预览：{err}"));
                            ui.label("可打开所在文件夹用系统看图工具查看。");
                        }
                        None => {
                            ui.spinner();
                            ui.label("正在后台解码…");
                        }
                    }
                }
            });
        if !open {
            self.results.large_asset = None;
        }
    }

    fn draw_compare_window(&mut self, ctx: &egui::Context) {
        let Some((table_name, group_id)) = self.results.compare_group.clone() else {
            return;
        };
        let Some(group) = self.find_group(&table_name, group_id) else {
            self.results.compare_group = None;
            return;
        };
        let keep = group
            .members
            .iter()
            .find(|member| member.is_recommended_keep)
            .or_else(|| group.members.first())
            .cloned();
        let compare = self
            .results
            .compare_file_id
            .and_then(|file_id| {
                group
                    .members
                    .iter()
                    .find(|member| member.file_id == file_id)
            })
            .or_else(|| group.members.get(1))
            .or_else(|| group.members.first())
            .cloned();
        let mut open = true;
        egui::Window::new("对比查看")
            .open(&mut open)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("上一组").clicked() {
                        self.step_compare_group(-1);
                    }
                    ui.label(format!("{} #{}", group.table_name, group.id));
                    if ui.button("下一组").clicked() {
                        self.step_compare_group(1);
                    }
                });
                ui.separator();
                ui.horizontal(|ui| {
                    if let Some(asset) = &keep {
                        self.draw_compare_side(ctx, ui, "建议保留", asset);
                    }
                    if let Some(asset) = &compare {
                        self.draw_compare_side(ctx, ui, "当前对比", asset);
                    }
                });
                ui.separator();
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for member in &group.members {
                            if ui
                                .selectable_label(
                                    self.results.compare_file_id == Some(member.file_id),
                                    &member.file_name,
                                )
                                .clicked()
                            {
                                self.results.compare_file_id = Some(member.file_id);
                            }
                        }
                    });
                });
            });
        if !open {
            self.results.compare_group = None;
        }
    }

    fn draw_compare_side(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        title: &str,
        asset: &CleanupAsset,
    ) {
        ui.vertical(|ui| {
            ui.set_width(360.0);
            ui.strong(title);
            match self.asset_texture(ctx, asset, 384) {
                Some(texture) => {
                    ui.image((texture.id(), egui::vec2(320.0, 320.0)));
                }
                None => {
                    let key = texture_key(asset, 384);
                    let failed = self.results.thumbnail_errors.contains_key(&key);
                    draw_thumbnail_placeholder(ui, 320.0, failed);
                }
            }
            ui.label(&asset.file_name);
            ui.label(format!(
                "{} / {} / {}",
                asset.asset_type,
                format_resolution(asset.width, asset.height),
                format_bytes(asset.file_size)
            ));
            if let Some(similarity) = asset.similarity {
                ui.label(format!("AI特征: {:.3}", similarity));
            }
            if let Some(distance) = asset.distance {
                ui.label(format!("pHash距离: {}", distance));
            }
        });
    }

    fn find_group(&self, table_name: &str, group_id: i64) -> Option<CleanupGroup> {
        let groups = if table_name == "duplicate_groups" {
            &self.cleanup_results.duplicate_groups
        } else {
            &self.cleanup_results.similarity_groups
        };
        groups.iter().find(|group| group.id == group_id).cloned()
    }

    fn step_compare_group(&mut self, delta: isize) {
        let Some((table_name, group_id)) = self.results.compare_group.clone() else {
            return;
        };
        let groups = if table_name == "duplicate_groups" {
            &self.cleanup_results.duplicate_groups
        } else {
            &self.cleanup_results.similarity_groups
        };
        if groups.is_empty() {
            return;
        }
        let current = groups
            .iter()
            .position(|group| group.id == group_id)
            .unwrap_or(0);
        let next = (current as isize + delta).rem_euclid(groups.len() as isize) as usize;
        self.results.compare_group = Some((table_name, groups[next].id));
        self.results.compare_file_id = groups[next]
            .members
            .iter()
            .find(|member| !member.is_recommended_keep)
            .or_else(|| groups[next].members.first())
            .map(|member| member.file_id);
    }

    /// Returns a texture only if one is already resident. Otherwise it queues a
    /// background decode and returns `None` so the caller can draw a
    /// placeholder. This function performs no image decoding of its own; that
    /// is the whole point of the change.
    fn asset_texture(
        &mut self,
        ctx: &egui::Context,
        asset: &CleanupAsset,
        size: u32,
    ) -> Option<egui::TextureHandle> {
        let key = texture_key(asset, size);
        if let Some(texture) = self.texture_cache.get(&key).cloned() {
            self.texture_budget.touch(&key);
            return Some(texture);
        }
        if asset.media_type != "IMAGE" && asset.asset_type != "LIVE_PHOTO" {
            // A flat colour block costs microseconds; there is nothing to
            // decode and nothing to send to a worker.
            let image = placeholder_image(size, [70, 76, 86]);
            let texture = ctx.load_texture(key.clone(), image, egui::TextureOptions::LINEAR);
            self.insert_texture(key, texture.clone(), size);
            return Some(texture);
        }
        if self.results.thumbnail_errors.contains_key(&key) {
            return None;
        }
        self.thumbnails.request(&key, &asset_path(asset), size);
        None
    }

    fn insert_texture(&mut self, key: String, texture: egui::TextureHandle, size: u32) {
        let bytes = (size as u64) * (size as u64) * 4;
        for evicted in self.texture_budget.insert(key.clone(), bytes) {
            self.texture_cache.remove(&evicted);
        }
        self.texture_cache.insert(key, texture);
    }

    fn drain_thumbnails(&mut self, ctx: &egui::Context) {
        for ready in self.thumbnails.poll() {
            match ready.result {
                Ok(rgb) => {
                    let side = ready.size as usize;
                    if rgb.len() != side * side * 3 {
                        self.results
                            .thumbnail_errors
                            .insert(ready.key, format!("解码结果尺寸异常：{} bytes", rgb.len()));
                        continue;
                    }
                    let image = egui::ColorImage::from_rgb([side, side], &rgb);
                    let texture =
                        ctx.load_texture(ready.key.clone(), image, egui::TextureOptions::LINEAR);
                    self.insert_texture(ready.key, texture, ready.size);
                }
                Err(err) => {
                    self.results.thumbnail_errors.insert(ready.key, err);
                }
            }
        }
    }

    fn stage_pending_assets(&mut self) {
        let pending: Vec<_> = self.results.pending_assets.values().cloned().collect();
        if pending.is_empty() {
            self.results.message = "待删除列表为空。".to_string();
            return;
        }
        let db = match Database::open(&self.paths) {
            Ok(db) => db,
            Err(err) => {
                self.results.message = format!("无法打开数据库：{err}");
                return;
            }
        };
        let batch_dir = self
            .paths
            .root
            .join("PhotoCleaner_待删除")
            .join(chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string());
        if let Err(err) = fs::create_dir_all(&batch_dir) {
            self.results.message = format!("无法创建待删除文件夹：{err}");
            return;
        }
        let mut moved_assets = 0usize;
        let mut errors = Vec::new();
        for asset in pending {
            let components = db.asset_file_components(asset.asset_id).unwrap_or_default();
            let components = if components.is_empty() {
                vec![crate::database::AssetFileComponent {
                    file_id: asset.file_id,
                    library_root: asset.library_root.clone(),
                    relative_path: asset.relative_path.clone(),
                    file_name: asset.file_name.clone(),
                }]
            } else {
                components
            };
            let mut moved_any = false;
            for component in components {
                let source = Path::new(&component.library_root).join(&component.relative_path);
                if !source.exists() {
                    errors.push(format!("源文件不存在：{}", source.display()));
                    continue;
                }
                let destination = unique_destination(&batch_dir, &component.file_name);
                if let Err(err) = move_file_safely(&source, &destination) {
                    errors.push(format!("移动失败：{} ({err})", source.display()));
                    continue;
                }
                let _ = db.record_move_operation(
                    component.file_id,
                    &source.display().to_string(),
                    &destination.display().to_string(),
                );
                moved_any = true;
            }
            if moved_any {
                moved_assets += 1;
                self.results.pending_assets.remove(&asset.asset_id);
            }
        }
        self.results.message = if errors.is_empty() {
            format!("已安全移动 {} 个资产到待删除文件夹。", moved_assets)
        } else {
            format!("已移动 {} 个资产，{} 个问题。", moved_assets, errors.len())
        };
    }

    fn undo_latest_move(&mut self) {
        let db = match Database::open(&self.paths) {
            Ok(db) => db,
            Err(err) => {
                self.results.message = format!("无法打开数据库：{err}");
                return;
            }
        };
        let operations = db.latest_move_operations(200).unwrap_or_default();
        if operations.is_empty() {
            self.results.message = "没有可撤销的移动记录。".to_string();
            return;
        }
        let mut restored = 0usize;
        for operation in operations {
            if restore_operation(&operation) {
                let _ = db.mark_operation_undone(operation.id);
                restored += 1;
            }
        }
        self.results.message = format!("已还原 {} 个文件。", restored);
    }

    fn draw_ai_status(&mut self, ui: &mut egui::Ui) {
        ui.heading("AI状态");
        ui.label(format!("AI模型：{}", self.ai_status.model_name));
        ui.label(format!(
            "模型状态：{}",
            if self.ai_status.model_loaded {
                "已加载"
            } else if self.ai_status.model_exists {
                "存在但未加载"
            } else {
                "缺失"
            }
        ));
        ui.label("Embedding：384维");
        ui.label(format!("推理设备：{}", self.ai_status.device));
        ui.label(format!(
            "CUDA状态：{}",
            if self.ai_status.cuda_available {
                "可用"
            } else {
                "不可用，不影响CPU深度扫描"
            }
        ));
        ui.label(&self.ai_status.detail);
        ui.horizontal(|ui| {
            if ui.button("刷新AI状态").clicked() {
                self.ai_status = crate::embedding::environment_check(&self.paths);
            }
            if ui.button("测试AI").clicked() {
                self.ai_test_result = Some(crate::embedding::test_ai(&self.paths));
                self.ai_status = crate::embedding::environment_check(&self.paths);
            }
        });
        if let Some(result) = &self.ai_test_result {
            ui.label(format!(
                "模型加载{}",
                if result.success { "成功" } else { "失败" }
            ));
            ui.label(format!("设备：{}", result.device));
            ui.label(format!("单次推理耗时：{} ms", result.elapsed_ms));
            ui.label(format!("输出维度：{}", result.output_dim));
            ui.label(format!(
                "NaN / Inf：{}",
                if result.has_nan_or_inf {
                    "存在"
                } else {
                    "无"
                }
            ));
            ui.label(&result.message);
        }
    }

    fn draw_progress(&mut self, ui: &mut egui::Ui) {
        let Some(progress) = &self.progress else {
            ui.heading("本次扫描");
            ui.label("尚未运行扫描");
            return;
        };

        ui.heading(if progress.stage == ScanStage::Done {
            "扫描完成"
        } else {
            "扫描媒体"
        });

        if progress.total_known && progress.total > 0 {
            let total_fraction = progress.completed as f32 / progress.total as f32;
            ui.add(
                egui::ProgressBar::new(total_fraction.clamp(0.0, 1.0))
                    .show_percentage()
                    .text(format!("{} / {}", progress.completed, progress.total)),
            );
        } else {
            ui.add(
                egui::ProgressBar::new(0.0)
                    .animate(true)
                    .text("正在查找媒体..."),
            );
            ui.label(format!("已发现：{}", progress.discovered));
        }

        ui.add_space(8.0);
        ui.label(format!("当前阶段：{}", progress.stage.label()));
        if progress.stage_total > 0 {
            let stage_fraction = progress.stage_completed as f32 / progress.stage_total as f32;
            ui.add(
                egui::ProgressBar::new(stage_fraction.clamp(0.0, 1.0))
                    .show_percentage()
                    .text(format!(
                        "{} / {}",
                        progress.stage_completed, progress.stage_total
                    )),
            );
        } else {
            ui.add(
                egui::ProgressBar::new(0.0)
                    .animate(true)
                    .text(&progress.activity),
            );
        }

        ui.add_space(8.0);
        egui::Grid::new("scan_stats").num_columns(4).show(ui, |ui| {
            ui.label(format!("发现媒体：{}", progress.discovered));
            ui.label(format!("已完成：{}", progress.completed));
            ui.label(format!("处理中：{}", progress.processing));
            ui.label(format!("新增：{}", progress.new_files));
            ui.end_row();
            ui.label(format!("已更新：{}", progress.updated_files));
            ui.label(format!("复用：{}", progress.reused_files));
            ui.label(format!("不支持：{}", progress.unsupported_files));
            ui.label(format!("读取失败：{}", progress.failed_files));
            ui.end_row();
        });

        let speed = if progress.stage == ScanStage::Done {
            "-".to_string()
        } else if progress.throughput > 0.0 {
            format!("{:.1} {}", progress.throughput, progress.throughput_unit)
        } else {
            "计算中...".to_string()
        };
        let eta = if progress.stage == ScanStage::Done {
            "完成".to_string()
        } else {
            progress
                .eta
                .map(format_duration)
                .unwrap_or_else(|| "计算中...".to_string())
        };
        ui.label(format!("速度：{speed}"));
        ui.label(format!("预计剩余：{eta}"));
        ui.label(format!("当前任务：{}", progress.activity));

        ui.collapsing("性能详情", |ui| {
            ui.label(format!(
                "CPU threads：{}",
                self.settings.resolved_cpu_threads()
            ));
            ui.label(format!(
                "Workers：{} / {}",
                progress.active_workers, progress.worker_count
            ));
            ui.label(format!("Metadata Queue：{}", progress.metadata_queue_len));
            ui.label(format!("Decode Queue：{}", progress.decode_queue_len));
            ui.label(format!("DB Queue：{}", progress.db_queue_len));
        });
    }
}

/// How many groups are laid out per page of the results wall.
const GROUPS_PER_PAGE: usize = 20;

fn texture_key(asset: &CleanupAsset, size: u32) -> String {
    format!("{}:{}:{}", asset.file_id, asset.relative_path, size)
}

fn draw_thumbnail_placeholder(ui: &mut egui::Ui, side: f32, failed: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click());
    ui.painter()
        .rect_filled(rect, 2.0, egui::Color32::from_rgb(38, 42, 50));
    let (text, color) = if failed {
        ("无法预览", egui::Color32::from_rgb(190, 120, 110))
    } else {
        ("载入中…", egui::Color32::from_rgb(150, 152, 162))
    };
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional((side * 0.11).clamp(11.0, 18.0)),
        color,
    );
    response
}

fn group_key(group: &CleanupGroup) -> String {
    format!("{}:{}", group.table_name, group.id)
}

/// Classification labels, as written by `database::pair_kind_label`.
const EXACT_DUPLICATE: &str = "EXACT_DUPLICATE";
const NEAR_DUPLICATE: &str = "NEAR_DUPLICATE";
const BURST_SIMILAR: &str = "BURST_SIMILAR";
const VISUALLY_SIMILAR: &str = "VISUALLY_SIMILAR";

fn kind_label(kind: &str) -> &'static str {
    match kind {
        EXACT_DUPLICATE => "完全重复",
        NEAR_DUPLICATE => "近重复",
        BURST_SIMILAR => "连拍",
        VISUALLY_SIMILAR => "相似照片",
        _ => "分组",
    }
}

fn evidence_text(group: &CleanupGroup) -> String {
    if group.kind == EXACT_DUPLICATE {
        return "依据：SHA-256完全一致，删除副本不会丢失任何画面".to_string();
    }
    let mut parts = Vec::new();
    if let Some(similarity) = group.members.iter().filter_map(|m| m.similarity).next() {
        parts.push(format!("AI特征: {:.3}", similarity));
    }
    if let Some(distance) = group.members.iter().filter_map(|m| m.distance).next() {
        parts.push(format!("pHash距离: {}", distance));
    }
    // `group.kind` is a String; the constants are &str, so the match has to be
    // on a &str or every arm is read as a fresh binding instead of a pattern.
    let caution = match group.kind.as_str() {
        NEAR_DUPLICATE => "近似判断，不是同一份文件，默认不勾选删除",
        BURST_SIMILAR => "连拍序列，画面本就相近，默认不勾选删除",
        VISUALLY_SIMILAR => "仅视觉相似，缺少重复证据，默认不勾选删除",
        _ => "默认不勾选删除",
    };
    if parts.is_empty() {
        format!("依据：数据库已有相似分组（{caution}）")
    } else {
        format!("依据：{}（{}）", parts.join("，"), caution)
    }
}

fn asset_path(asset: &CleanupAsset) -> PathBuf {
    Path::new(&asset.library_root).join(&asset.relative_path)
}

fn open_in_explorer(path: &Path) {
    let _ = Command::new("explorer").arg("/select,").arg(path).spawn();
}

fn placeholder_image(size: u32, color: [u8; 3]) -> egui::ColorImage {
    let mut pixels = Vec::with_capacity((size * size * 3) as usize);
    for _ in 0..(size * size) {
        pixels.extend_from_slice(&color);
    }
    egui::ColorImage::from_rgb([size as usize, size as usize], &pixels)
}

fn unique_destination(folder: &Path, file_name: &str) -> PathBuf {
    let mut destination = folder.join(file_name);
    if !destination.exists() {
        return destination;
    }
    let stem = Path::new(file_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("file");
    let extension = Path::new(file_name)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    for index in 1..10_000 {
        let candidate = if extension.is_empty() {
            folder.join(format!("{stem}_{index}"))
        } else {
            folder.join(format!("{stem}_{index}.{extension}"))
        };
        if !candidate.exists() {
            destination = candidate;
            break;
        }
    }
    destination
}

fn restore_operation(operation: &MoveOperation) -> bool {
    let source = PathBuf::from(&operation.source_path);
    let destination = PathBuf::from(&operation.destination_path);
    if !destination.exists() || source.exists() {
        return false;
    }
    if let Some(parent) = source.parent() {
        let _ = fs::create_dir_all(parent);
    }
    move_file_safely(&destination, &source).is_ok()
}

fn move_file_safely(source: &Path, destination: &Path) -> std::io::Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    match fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(rename_err) => {
            fs::copy(source, destination)?;
            match fs::remove_file(source) {
                Ok(()) => Ok(()),
                Err(remove_err) => {
                    let _ = fs::remove_file(destination);
                    Err(std::io::Error::new(
                        remove_err.kind(),
                        format!("rename failed: {rename_err}; remove source failed: {remove_err}"),
                    ))
                }
            }
        }
    }
}

fn format_resolution(width: Option<i64>, height: Option<i64>) -> String {
    match (width, height) {
        (Some(width), Some(height)) if width > 0 && height > 0 => format!("{width} x {height}"),
        _ => "未知分辨率".to_string(),
    }
}

fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let value = bytes as f64;
    if value >= GB {
        format!("{:.2} GB", value / GB)
    } else if value >= MB {
        format!("{:.1} MB", value / MB)
    } else if value >= KB {
        format!("{:.1} KB", value / KB)
    } else {
        format!("{bytes} B")
    }
}

fn short_time(value: &str) -> String {
    value.chars().take(19).collect()
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    format!("{:02}:{:02}", secs / 60, secs % 60)
}

fn format_ms(ms: u128) -> String {
    format_duration(Duration::from_millis(ms as u64))
}
