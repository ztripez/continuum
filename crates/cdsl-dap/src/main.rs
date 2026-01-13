use cdsl_dap::server::DapServer;
use tokio::io::{stdin, stdout};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing to file
    let file_appender = tracing_appender::rolling::never("/tmp", "cdsl-dap.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")),
        )
        .init();

    tracing::info!("CDSL DAP starting...");

    let server = DapServer::new();
    server.run(stdin(), stdout()).await?;

    Ok(())
}
