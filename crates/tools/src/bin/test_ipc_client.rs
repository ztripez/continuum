//! Simple IPC test client for checkpoint commands

use continuum_tools::ipc_protocol::*;
use std::path::PathBuf;
use tokio::io::BufReader;
use tokio::net::UnixStream;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let socket = PathBuf::from("/tmp/continuum-test.sock");

    println!("=== IPC Checkpoint Test Client ===\n");

    // Test 1: Get status
    println!("[1] Getting initial status...");
    let resp = send_command(&socket, IpcCommand::Status).await?;
    if let Some(IpcResponsePayload::Status(status)) = &resp.payload {
        println!(
            "  Tick: {}, Sim Time: {:.3}, Era: {}, Running: {}\n",
            status.tick, status.sim_time, status.era, status.running
        );
    }

    // Test 2: Run some steps
    println!("[2] Running 5 steps...");
    for i in 1..=5 {
        let resp = send_command(&socket, IpcCommand::Step { count: 1 }).await?;
        if let Some(IpcResponsePayload::Status(status)) = &resp.payload {
            println!(
                "  Step {}: tick={}, sim_time={:.3}",
                i, status.tick, status.sim_time
            );
        }
    }
    println!();

    // Test 3: Request checkpoint
    println!("[3] Requesting checkpoint...");
    let ckpt_path = format!(
        "examples/entity-test/test_checkpoint_{}.ckpt",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );
    let resp = send_command(
        &socket,
        IpcCommand::CheckpointRequest {
            path: Some(ckpt_path.clone()),
        },
    )
    .await?;
    if let Some(IpcResponsePayload::CheckpointRequest(ckpt)) = &resp.payload {
        println!(
            "  Checkpoint created: tick={}, path={}\n",
            ckpt.tick, ckpt.path
        );
    }

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Test 4: List checkpoints
    println!("[4] Listing checkpoints...");
    let resp = send_command(
        &socket,
        IpcCommand::CheckpointList {
            dir: "examples/entity-test".to_string(),
        },
    )
    .await?;
    if let Some(IpcResponsePayload::CheckpointList(list)) = &resp.payload {
        println!("  Found {} checkpoint(s):", list.checkpoints.len());
        for ckpt in &list.checkpoints {
            println!(
                "    - tick={}, path={}, size={}B",
                ckpt.tick, ckpt.path, ckpt.size_bytes
            );
        }
    }
    println!();

    // Test 5: Run more steps
    println!("[5] Running 3 more steps...");
    for i in 1..=3 {
        let resp = send_command(&socket, IpcCommand::Step { count: 1 }).await?;
        if let Some(IpcResponsePayload::Status(status)) = &resp.payload {
            println!(
                "  Step {}: tick={}, sim_time={:.3}",
                i + 5,
                status.tick,
                status.sim_time
            );
        }
    }
    println!();

    // Test 6: Stop simulation
    println!("[6] Stopping simulation (if running)...");
    send_command(&socket, IpcCommand::Stop).await?;
    println!();

    // Test 7: Resume from checkpoint
    println!("[7] Resuming from checkpoint: {}", ckpt_path);
    let resp = send_command(
        &socket,
        IpcCommand::CheckpointResume {
            path: ckpt_path.clone(),
            force: false,
        },
    )
    .await?;
    if let Some(IpcResponsePayload::CheckpointResume(resume)) = &resp.payload {
        println!(
            "  Resumed: tick={}, sim_time={:.3}, era={}\n",
            resume.tick, resume.sim_time, resume.era
        );
    }

    // Test 8: Verify state
    println!("[8] Getting status after resume...");
    let resp = send_command(&socket, IpcCommand::Status).await?;
    if let Some(IpcResponsePayload::Status(status)) = &resp.payload {
        println!(
            "  Tick: {}, Sim Time: {:.3}, Era: {}\n",
            status.tick, status.sim_time, status.era
        );
    }

    // Test 9: Remove checkpoint
    println!("[9] Removing checkpoint: {}", ckpt_path);
    let resp = send_command(
        &socket,
        IpcCommand::CheckpointRemove {
            path: ckpt_path.clone(),
        },
    )
    .await?;
    if resp.ok {
        println!("  Checkpoint removed successfully\n");
    }

    println!("=== Test Complete ===");
    Ok(())
}

async fn send_command(socket_path: &PathBuf, command: IpcCommand) -> anyhow::Result<IpcResponse> {
    let stream = UnixStream::connect(socket_path).await?;
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    let request = IpcRequest {
        id: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        command,
    };

    write_frame(&mut writer, &request).await?;
    let frame: IpcFrame = read_frame(&mut reader).await?;

    match frame {
        IpcFrame::Response(response) => Ok(response),
        IpcFrame::Event(_) => anyhow::bail!("Received event instead of response"),
    }
}
