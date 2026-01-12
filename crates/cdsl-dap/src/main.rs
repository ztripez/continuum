use cdsl_dap::adapter::ContinuumDebugAdapter;
use tokio::io::{stdin, stdout, BufReader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _reader = BufReader::new(stdin());
    let _writer = stdout();

    let _adapter = ContinuumDebugAdapter::new();

    // TODO: Wire up DAP server implementation

    Ok(())
}
