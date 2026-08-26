use std::path::PathBuf;
use std::thread;

use crossbeam_channel::{unbounded, Receiver, Sender};

use crate::database::Database;
use crate::logging;
use crate::paths::PortablePaths;
use crate::scanner::{self, ScanMode, ScanOutcome, ScanProgress};

#[derive(Clone, Debug)]
pub enum TaskCommand {
    Scan { root: PathBuf, mode: ScanMode },
}

#[derive(Clone, Debug)]
pub enum TaskEvent {
    ScanStarted { root: PathBuf, mode: ScanMode },
    ScanProgress(ScanProgress),
    ScanFinished { outcome: ScanOutcome },
    Failed(String),
}

pub struct TaskRunner {
    sender: Sender<TaskCommand>,
    receiver: Receiver<TaskEvent>,
}

impl TaskRunner {
    pub fn start(paths: PortablePaths) -> Self {
        let (cmd_tx, cmd_rx) = unbounded();
        let (event_tx, event_rx) = unbounded();

        thread::spawn(move || {
            while let Ok(command) = cmd_rx.recv() {
                match command {
                    TaskCommand::Scan { root, mode } => {
                        let _ = event_tx.send(TaskEvent::ScanStarted {
                            root: root.clone(),
                            mode,
                        });
                        let tx = event_tx.clone();
                        let result = scanner::run_pipeline(&paths, root, mode, move |progress| {
                            let _ = tx.send(TaskEvent::ScanProgress(progress));
                        });
                        match result {
                            Ok(outcome) => {
                                logging::info(format!(
                                    "scan complete: {} records written",
                                    outcome.written
                                ));
                                let _ = event_tx.send(TaskEvent::ScanFinished { outcome });
                            }
                            Err(err) => {
                                logging::error(format!("scan failed: {err:#}"));
                                let _ = event_tx.send(TaskEvent::Failed(format!("{err:#}")));
                            }
                        }
                    }
                }
            }
        });

        Self {
            sender: cmd_tx,
            receiver: event_rx,
        }
    }

    pub fn sender(&self) -> Sender<TaskCommand> {
        self.sender.clone()
    }

    pub fn drain_events(&self) -> Vec<TaskEvent> {
        self.receiver.try_iter().collect()
    }
}

pub fn refresh_counts(
    paths: &PortablePaths,
) -> (Vec<crate::database::Library>, crate::database::MediaCounts) {
    match Database::open(paths) {
        Ok(db) => (
            db.list_libraries().unwrap_or_default(),
            db.media_counts().unwrap_or_default(),
        ),
        Err(_) => (Vec::new(), Default::default()),
    }
}
