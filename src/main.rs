//! Boot: capture the local UTC offset (before the runtime spawns
//! threads), load `.env`, read config, open the data dir, build the
//! Responses client and the tool registry, restore the agent, then run
//! the status web UI, the schedulers, and the Discord endpoint.

use std::sync::Arc;

use tracing::{error, info};

use pester::agent::{spawn_idle_compactor, spawn_midnight_distiller, Agent, AgentSettings, SharedStores};
use pester::config::AppConfig;
use pester::data::DataDir;
use pester::daynote::DayNoteStore;
use pester::reminders::{spawn_scheduler, ReminderStore};
use pester::responses::ResponsesClient;
use pester::status::Status;
use pester::tools::fs::{ListFiles, ReadFile, WriteFile};
use pester::tools::memory::UpdateDayMemory;
use pester::tools::remind::{CreateReminder, DeleteReminder, ListReminders, SatisfyReminder};
use pester::tools::time::GetTime;
use pester::tools::Registry;
use pester::{discord, web};

fn main() {
    // Must run while the process is still single-threaded: the OS local
    // offset lookup refuses once threads exist.
    pester::data::init_local_offset();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("building the tokio runtime")
        .block_on(async_main());
}

async fn async_main() {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "pester=info,serenity=warn".into()),
        )
        .init();

    let config = match AppConfig::from_env() {
        Ok(c) => c,
        Err(e) => {
            error!("configuration error: {e}");
            std::process::exit(1);
        }
    };

    let data = match DataDir::open(&config.data_dir) {
        Ok(d) => d,
        Err(e) => {
            error!("data directory error: {e}");
            std::process::exit(1);
        }
    };
    info!(root = %data.root().display(), "data directory ready");

    let files_root = data.files_dir().to_path_buf();
    let reminders = Arc::new(ReminderStore::new(data.reminders_path()));
    let daynotes = Arc::new(DayNoteStore::new(data.daynotes_path()));
    let registry = Registry::new(vec![
        Arc::new(ReadFile::new(&files_root)),
        Arc::new(WriteFile::new(&files_root)),
        Arc::new(ListFiles::new(&files_root)),
        Arc::new(GetTime),
        Arc::new(CreateReminder::new(Arc::clone(&reminders))),
        Arc::new(ListReminders::new(Arc::clone(&reminders))),
        Arc::new(DeleteReminder::new(Arc::clone(&reminders))),
        Arc::new(SatisfyReminder::new(Arc::clone(&reminders))),
        Arc::new(UpdateDayMemory::new(Arc::clone(&daynotes))),
    ]);

    let llm = Arc::new(ResponsesClient::new(config.provider.clone()));
    let status = Arc::new(Status::new(&config.provider.model));
    let (compact_tx, compact_rx) = tokio::sync::mpsc::channel(8);
    let agent = Arc::new(Agent::new(
        llm,
        data.clone(),
        registry,
        Arc::clone(&status),
        AgentSettings {
            model: config.provider.model.clone(),
            reasoning: config.reasoning,
            compact_tokens: config.compact_tokens,
        },
        SharedStores {
            daynotes,
            reminders: Arc::clone(&reminders),
        },
        compact_tx,
    ));
    spawn_idle_compactor(Arc::clone(&agent), config.idle_compact_mins);
    spawn_midnight_distiller(Arc::clone(&agent));

    let (fire_tx, fire_rx) = tokio::sync::mpsc::channel(32);
    spawn_scheduler(reminders, fire_tx, Arc::clone(&status));

    let web_status = Arc::clone(&status);
    let web_addr = config.web_addr.clone();
    tokio::spawn(async move {
        if let Err(e) = web::serve(web_status, &web_addr).await {
            error!("web ui failed: {e}");
        }
    });

    info!(
        allowed_user = ?config.discord_user,
        model = %config.provider.model,
        "starting discord DM endpoint"
    );
    if let Err(e) = discord::run(
        config.discord_token,
        config.discord_user,
        config.user_aliases,
        agent,
        status,
        data,
        compact_rx,
        fire_rx,
    )
    .await
    {
        error!("discord client failed: {e}");
        std::process::exit(1);
    }
}
