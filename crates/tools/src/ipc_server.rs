use std::path::Path;
use tokio::net::UnixListener;
use crate::world_api::framing::read_message;
use crate::run_world_intent::RunWorldIntent;
use tracing::{info, error, debug};

/// A simple IPC server that executes a world intent.
pub struct WorldIpcServer {
    intent: RunWorldIntent,
}

impl WorldIpcServer {
    pub fn new(intent: RunWorldIntent) -> Self {
        Self { intent }
    }

    pub async fn run(self, socket_path: &Path) -> Result<(), std::io::Error> {
        if socket_path.exists() {
            std::fs::remove_file(socket_path)?;
        }

        let listener = UnixListener::bind(socket_path)?;
        info!("World IPC server listening on {}", socket_path.display());

        loop {
            match listener.accept().await {
                Ok((mut stream, _)) => {
                    info!("Client connected to World IPC");
                    
                    // Simple IPC loop: echo for now, would integrate with intent.execute()
                    loop {
                        match read_message(&mut stream).await {
                            Ok(msg) => {
                                debug!("Received IPC message: {:?}", msg);
                                // Handle message...
                            }
                            Err(e) => {
                                error!("IPC read error: {}", e);
                                break;
                            }
                        }
                    }
                }
                Err(e) => error!("Failed to accept IPC connection: {}", e),
            }
        }
    }
}
