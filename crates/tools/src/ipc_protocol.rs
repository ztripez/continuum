use std::collections::BTreeMap;

use anyhow::Context;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use continuum_foundation::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum JsonValue {
    Scalar(f64),
    Boolean(bool),
    Integer(i64),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
    Vec4([f64; 4]),
    Quat([f64; 4]),
    Struct(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub fn to_value(&self) -> Value {
        match self {
            JsonValue::Scalar(value) => Value::Scalar(*value),
            JsonValue::Boolean(value) => Value::Boolean(*value),
            JsonValue::Integer(value) => Value::Integer(*value),
            JsonValue::Vec2(value) => Value::Vec2(*value),
            JsonValue::Vec3(value) => Value::Vec3(*value),
            JsonValue::Vec4(value) => Value::Vec4(*value),
            JsonValue::Quat(value) => Value::Quat(*value),
            JsonValue::Struct(fields) => {
                let mut map = serde_json::Map::new();
                for (key, value) in fields {
                    map.insert(key.clone(), serde_json::to_value(value.to_value()).unwrap());
                }
                Value::Data(serde_json::Value::Object(map))
            }
        }
    }

    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Scalar(v) => JsonValue::Scalar(*v),
            Value::Boolean(v) => JsonValue::Boolean(*v),
            Value::Integer(v) => JsonValue::Integer(*v),
            Value::Vec2(v) => JsonValue::Vec2(*v),
            Value::Vec3(v) => JsonValue::Vec3(*v),
            Value::Vec4(v) => JsonValue::Vec4(*v),
            Value::Quat(v) => JsonValue::Quat(*v),
            Value::Data(v) => {
                if let Some(obj) = v.as_object() {
                    let mut fields = BTreeMap::new();
                    for (key, val) in obj {
                        // This is a bit recursive/inefficient but it works
                        let continuum_val: Value = serde_json::from_value(val.clone()).unwrap();
                        fields.insert(key.clone(), JsonValue::from_value(&continuum_val));
                    }
                    JsonValue::Struct(fields)
                } else {
                    JsonValue::Scalar(0.0) // Fallback
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRequest {
    pub id: u64,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonResponse {
    pub id: u64,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StepRequest {
    pub count: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RunRequest {
    pub count: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FieldHistoryRequest {
    pub field_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FieldQueryRequest {
    pub field_id: String,
    pub position: [f64; 3],
    pub time: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FieldQueryBatchRequest {
    pub field_id: String,
    pub positions: Vec<[f64; 3]>,
    pub tick: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FieldLatestRequest {
    pub field_id: String,
    pub position: Option<[f64; 3]>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FieldTileRequest {
    pub field_id: String,
    pub tile: TileAddress,
    pub tick: u64,
    pub positions: Vec<[f64; 3]>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlaybackSetRequest {
    pub lag_ticks: f64,
    pub speed: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlaybackSeekRequest {
    pub time: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlaybackQueryRequest {
    pub field_id: String,
    pub position: [f64; 3],
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImpulseEmitRequest {
    pub impulse_id: String,
    pub payload: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcRequest {
    pub id: u64,
    pub command: IpcCommand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcCommand {
    Status,
    Step {
        count: u64,
    },
    Run {
        count: Option<u64>,
    },
    Stop,
    FieldList,
    FieldHistory {
        field_id: String,
    },
    FieldQuery {
        field_id: String,
        position: [f64; 3],
        time: Option<f64>,
    },
    FieldQueryBatch {
        field_id: String,
        positions: Vec<[f64; 3]>,
        tick: Option<u64>,
    },
    FieldLatest {
        field_id: String,
        position: Option<[f64; 3]>,
    },
    FieldTile {
        field_id: String,
        tile: TileAddress,
        tick: u64,
        positions: Vec<[f64; 3]>,
    },
    PlaybackSet {
        lag_ticks: f64,
        speed: f64,
    },
    PlaybackSeek {
        time: f64,
    },
    PlaybackQuery {
        field_id: String,
        position: [f64; 3],
    },
    ChroniclePoll,
    ImpulseList,
    ImpulseEmit {
        impulse_id: String,
        payload: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileAddress {
    pub face: u8,
    pub lod: u8,
    pub morton: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcResponse {
    pub id: u64,
    pub ok: bool,
    pub payload: Option<IpcResponsePayload>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcResponsePayload {
    Status(StatusPayload),
    FieldList(FieldListPayload),
    FieldHistory(FieldHistoryPayload),
    FieldQuery(FieldQueryPayload),
    FieldQueryBatch(FieldQueryBatchPayload),
    FieldLatest(FieldLatestPayload),
    ChroniclePoll(ChroniclePollPayload),
    ImpulseList(ImpulseListPayload),
    ImpulseEmit(ImpulseEmitPayload),
    Playback(PlaybackPayload),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcEvent {
    Tick(TickEvent),
    Chronicle(ChronicleEvent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusPayload {
    pub tick: u64,
    pub era: String,
    pub sim_time: f64,
    pub dt: f64,
    pub phase: String,
    pub running: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldListPayload {
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldHistoryPayload {
    pub field_id: String,
    pub ticks: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldQueryPayload {
    pub field_id: String,
    pub value: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldQueryBatchPayload {
    pub field_id: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldLatestPayload {
    pub field_id: String,
    pub tick: u64,
    pub value: Option<JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChroniclePollPayload {
    pub events: Vec<ChronicleEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseInfo {
    pub id: String,
    pub payload_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseListPayload {
    pub impulses: Vec<ImpulseInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseEmitPayload {
    pub seq: u64,
    pub applied_tick: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybackPayload {
    pub time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickEvent {
    pub tick: u64,
    pub era: String,
    pub sim_time: f64,
    pub field_count: usize,
    pub event_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronicleEvent {
    pub chronicle_id: String,
    pub name: String,
    pub fields: Vec<(String, JsonValue)>,
    pub tick: u64,
    pub era: String,
    pub sim_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcFrame {
    Response(IpcResponse),
    Event(IpcEvent),
}

pub fn ipc_response_to_json(response: &IpcResponse) -> JsonResponse {
    let payload = response
        .payload
        .as_ref()
        .and_then(|p| serde_json::to_value(p).ok());
    JsonResponse {
        id: response.id,
        ok: response.ok,
        payload,
        error: response.error.clone(),
    }
}

pub fn json_request_to_ipc(request: JsonRequest) -> anyhow::Result<IpcRequest> {
    let command = match request.kind.as_str() {
        "status" => IpcCommand::Status,
        "step" => {
            let req: StepRequest = serde_json::from_value(request.payload)?;
            IpcCommand::Step {
                count: req.count.unwrap_or(1),
            }
        }
        "run" => {
            let req: RunRequest = serde_json::from_value(request.payload)?;
            IpcCommand::Run { count: req.count }
        }
        "stop" => IpcCommand::Stop,
        "field.list" => IpcCommand::FieldList,
        "field.history" => {
            let req: FieldHistoryRequest = serde_json::from_value(request.payload)?;
            IpcCommand::FieldHistory {
                field_id: req.field_id,
            }
        }
        "field.query" => {
            let req: FieldQueryRequest = serde_json::from_value(request.payload)?;
            IpcCommand::FieldQuery {
                field_id: req.field_id,
                position: req.position,
                time: req.time,
            }
        }
        "field.query_batch" => {
            let req: FieldQueryBatchRequest = serde_json::from_value(request.payload)?;
            IpcCommand::FieldQueryBatch {
                field_id: req.field_id,
                positions: req.positions,
                tick: req.tick,
            }
        }
        "field.latest" => {
            let req: FieldLatestRequest = serde_json::from_value(request.payload)?;
            IpcCommand::FieldLatest {
                field_id: req.field_id,
                position: req.position,
            }
        }
        "field.tile" => {
            let req: FieldTileRequest = serde_json::from_value(request.payload)?;
            IpcCommand::FieldTile {
                field_id: req.field_id,
                tile: req.tile,
                tick: req.tick,
                positions: req.positions,
            }
        }
        "playback.set" => {
            let req: PlaybackSetRequest = serde_json::from_value(request.payload)?;
            IpcCommand::PlaybackSet {
                lag_ticks: req.lag_ticks,
                speed: req.speed,
            }
        }
        "playback.seek" => {
            let req: PlaybackSeekRequest = serde_json::from_value(request.payload)?;
            IpcCommand::PlaybackSeek { time: req.time }
        }
        "playback.query" => {
            let req: PlaybackQueryRequest = serde_json::from_value(request.payload)?;
            IpcCommand::PlaybackQuery {
                field_id: req.field_id,
                position: req.position,
            }
        }
        "chronicle.poll" => IpcCommand::ChroniclePoll,
        "impulse.list" => IpcCommand::ImpulseList,
        "impulse.emit" => {
            let req: ImpulseEmitRequest = serde_json::from_value(request.payload)?;
            IpcCommand::ImpulseEmit {
                impulse_id: req.impulse_id,
                payload: req.payload.to_value(),
            }
        }
        _ => anyhow::bail!("unknown request type: {}", request.kind),
    };

    Ok(IpcRequest {
        id: request.id,
        command,
    })
}

pub fn ipc_event_to_json(event: &IpcEvent) -> JsonEvent {
    match event {
        IpcEvent::Tick(payload) => JsonEvent {
            kind: "tick".to_string(),
            payload: serde_json::to_value(payload).expect("tick serialize"),
        },
        IpcEvent::Chronicle(payload) => JsonEvent {
            kind: "chronicle.event".to_string(),
            payload: serde_json::to_value(payload).expect("chronicle serialize"),
        },
    }
}

pub async fn read_frame<R, T>(reader: &mut R) -> anyhow::Result<T>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    let len = reader.read_u32_le().await.context("read frame length")?;
    let mut buffer = vec![0u8; len as usize];
    reader
        .read_exact(&mut buffer)
        .await
        .context("read frame payload")?;
    let value = bincode::deserialize(&buffer).context("decode frame")?;
    Ok(value)
}

pub async fn write_frame<W, T>(writer: &mut W, value: &T) -> anyhow::Result<()>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let payload = bincode::serialize(value).context("encode frame")?;
    let len = payload.len() as u32;
    writer
        .write_u32_le(len)
        .await
        .context("write frame length")?;
    writer
        .write_all(&payload)
        .await
        .context("write frame payload")?;
    writer.flush().await.context("flush frame")?;
    Ok(())
}
