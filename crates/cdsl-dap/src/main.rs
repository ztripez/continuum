use cdsl_dap::server::DapServer;
use tokio::io::{stdin, stdout};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = DapServer::new();
    server.run(stdin(), stdout()).await?;

    Ok(())
}
