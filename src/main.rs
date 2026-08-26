mod ann;
mod app;
mod config;
mod database;
mod embedding;
mod file_ops;
mod grouping;
mod hardware;
mod hashing;
mod logging;
mod media_probe;
mod metadata;
mod paths;
mod perf;
mod phash;
mod quality;
mod scan_planner;
mod scanner;
mod tasks;
mod thumbnails;
mod ui;

fn main() -> eframe::Result<()> {
    let root = paths::PortablePaths::from_current_exe().unwrap_or_else(|err| {
        eprintln!("Failed to resolve portable paths: {err}");
        std::process::exit(1);
    });

    if let Err(err) = root.ensure_layout() {
        eprintln!("{err}");
    }

    logging::init(&root).ok();

    let arg1 = std::env::args().nth(1);
    if let Some(scan_root) = std::env::args().nth(2).filter(|_| {
        arg1.as_deref()
            .is_some_and(|arg| arg == "--scan-bench" || arg == "--scan-bench-deep")
    }) {
        let mode = if arg1.as_deref() == Some("--scan-bench-deep") {
            scanner::ScanMode::Deep
        } else {
            scanner::ScanMode::Standard
        };
        let outcome =
            scanner::run_pipeline(&root, std::path::PathBuf::from(scan_root), mode, |_| {})
                .unwrap_or_else(|err| {
                    eprintln!("{err:#}");
                    std::process::exit(1);
                });
        println!(
            "{}",
            serde_json::to_string_pretty(&outcome.summary).unwrap()
        );
        return Ok(());
    }
    if arg1.as_deref() == Some("--ai-test") {
        let result = embedding::test_ai(&root);
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        if !result.success {
            std::process::exit(1);
        }
        return Ok(());
    }
    if arg1.as_deref() == Some("--ai-bench") {
        let images = std::env::args()
            .nth(2)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(100)
            .max(1);
        let result = embedding::benchmark_generated(&root, images);
        write_cpu_performance_report(&root, &result).ok();
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        if !result.success {
            std::process::exit(1);
        }
        return Ok(());
    }
    if arg1.as_deref() == Some("--recognition-rebuild") {
        let Some(scan_root) = std::env::args().nth(2) else {
            eprintln!("Missing folder path");
            std::process::exit(1);
        };
        let result = scanner::rebuild_recognition_only(&root, std::path::PathBuf::from(scan_root))
            .unwrap_or_else(|err| {
                eprintln!("{err:#}");
                std::process::exit(1);
            });
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        return Ok(());
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 760.0]),
        ..Default::default()
    };

    eframe::run_native(
        "PhotoCleaner",
        options,
        Box::new(move |cc| {
            ui::fonts::install_chinese_fonts(&cc.egui_ctx);
            Box::new(app::PhotoCleanerApp::new(root.clone()))
        }),
    )
}

fn write_cpu_performance_report(
    root: &paths::PortablePaths,
    result: &embedding::AiBenchmarkResult,
) -> std::io::Result<()> {
    let text = format!(
        "# CPU Performance Report\n\n\
CPU: {}\n\n\
Physical cores: {}\n\n\
Logical processors: {}\n\n\
Previous threading model: scan workers used logical processors minus one, and every image worker could invoke ONNX inference directly. This allowed nested parallelism between scanner workers and ONNX Runtime CPU threads.\n\n\
Nested parallelism found: yes\n\n\
New threading model: DEEP mode uses a bounded AI inference queue, one AI coordinator, one ONNX Session, ORT_SEQUENTIAL execution, ORT_ENABLE_ALL graph optimization, and CPU intra-op threads based on physical cores.\n\n\
Batch behavior: requested batch is used when the ONNX model accepts it. If the model has a fixed batch=1 input shape, PhotoCleaner automatically falls back to single-image inference inside the same coordinator thread and keeps the single shared Session.\n\n\
AI threads: {}\n\n\
AI batch: {}\n\n\
Thread spinning: {}\n\n\
Benchmark images: {}\n\n\
Elapsed ms: {}\n\n\
Images/sec: {:.3}\n\n\
Output dimension: {}\n\n\
NaN/Inf: {}\n\n\
Status: {}\n\n\
CPU utilization: not sampled in this portable benchmark; throughput is the primary metric.\n\n\
Notes: no hardware settings were changed. No Ryzen Master, PBO, BIOS, Windows power plan, CPU affinity, or frequency lock was used.\n",
        result.topology.cpu_name,
        result.topology.physical_cores,
        result.topology.logical_processors,
        result.profile.cpu_threads,
        result.profile.batch_size,
        result.profile.thread_spinning,
        result.images,
        result.elapsed_ms,
        result.images_per_second,
        result.output_dim,
        result.has_nan_or_inf,
        result.message
    );
    std::fs::write(root.root.join("CPU_PERFORMANCE_REPORT.md"), text)
}
