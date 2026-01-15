use std::collections::BTreeMap;

use anyhow::Context;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use continuum_foundation::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JsonValue {
    Scalar(f64),
    Boolean(bool),
    Integer(i64),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
    Vec4([f64; 4]),
    Quat([f64; 4]),
    Mat2([f64; 4]),
    Mat3([f64; 9]),
    Mat4([f64; 16]),
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
            JsonValue::Mat2(value) => Value::Mat2(*value),
            JsonValue::Mat3(value) => Value::Mat3(*value),
            JsonValue::Mat4(value) => Value::Mat4(*value),
            JsonValue::Struct(fields) => {
                let items = fields
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_value()))
                    .collect();
                Value::Map(items)
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
            Value::Mat2(v) => JsonValue::Mat2(*v),
            Value::Mat3(v) => JsonValue::Mat3(*v),
            Value::Mat4(v) => JsonValue::Mat4(*v),
            Value::Map(v) => {
                let mut fields = BTreeMap::new();
                for (k, val) in v {
                    fields.insert(k.clone(), Self::from_value(val));
                }
                JsonValue::Struct(fields)
            }
            Value::Tensor(_) => {
                // Tensors are not yet supported in JSON serialization
                JsonValue::Struct(BTreeMap::new())
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
pub struct FieldDescribeRequest {
    pub field_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SignalDescribeRequest {
    pub signal_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StratumDescribeRequest {
    pub stratum_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EraDescribeRequest {
    pub era_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EntityDescribeRequest {
    pub entity_id: String,
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
    WorldInfo,
    SignalList,
    SignalDescribe {
        signal_id: String,
    },
    FieldList,
    FieldDescribe {
        field_id: String,
    },
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
    StratumList,
    StratumDescribe {
        stratum_id: String,
    },
    EraList,
    EraDescribe {
        era_id: String,
    },
    EntityList,
    EntityDescribe {
        entity_id: String,
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
    WorldInfo(WorldInfo),
    SignalList(SignalListPayload),
    SignalDescribe(SignalInfo),
    FieldList(FieldListPayload),
    FieldDescribe(FieldInfo),
    FieldHistory(FieldHistoryPayload),
    FieldQuery(FieldQueryPayload),
    FieldQueryBatch(FieldQueryBatchPayload),
    FieldLatest(FieldLatestPayload),
    ChroniclePoll(ChroniclePollPayload),
    ImpulseList(ImpulseListPayload),
    ImpulseEmit(ImpulseEmitPayload),
    Playback(PlaybackPayload),
    StratumList(StratumListPayload),
    StratumDescribe(StratumInfo),
    EraList(EraListPayload),
    EraDescribe(EraInfo),
    EntityList(EntityListPayload),
    EntityDescribe(EntityInfo),
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
pub struct FieldInfo {
    pub id: String,
    pub doc: Option<String>,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub value_type: String,
    pub unit: Option<String>,
    pub range: Option<(f64, f64)>,
    pub topology: String,
    pub stratum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalListPayload {
    pub signals: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalInfo {
    pub id: String,
    pub doc: Option<String>,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub value_type: String,
    pub unit: Option<String>,
    pub range: Option<(f64, f64)>,
    pub stratum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseInfo {
    pub id: String,
    pub doc: Option<String>,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub payload_type: String,
    pub unit: Option<String>,
    pub range: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldInfo {
    pub strata: Vec<StratumInfo>,
    pub eras: Vec<EraInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumListPayload {
    pub strata: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumInfo {
    pub id: String,
    pub doc: Option<String>,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub default_stride: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EraListPayload {
    pub eras: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EraInfo {
    pub id: String,
    pub doc: Option<String>,
    pub title: Option<String>,
    pub is_initial: bool,
    pub is_terminal: bool,
    pub dt_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityListPayload {
    pub entities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInfo {
    pub id: String,
    pub doc: Option<String>,
    pub count_bounds: Option<(u32, u32)>,
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

impl JsonValue {
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            JsonValue::Scalar(v) => {
                if v.is_finite() {
                    serde_json::json!(*v)
                } else {
                    serde_json::json!(0.0)
                }
            }
            JsonValue::Boolean(v) => serde_json::json!(*v),
            JsonValue::Integer(v) => serde_json::json!(*v),
            JsonValue::Vec2(v) => {
                let v = [
                    if v[0].is_finite() { v[0] } else { 0.0 },
                    if v[1].is_finite() { v[1] } else { 0.0 },
                ];
                serde_json::json!(v)
            }
            JsonValue::Vec3(v) => {
                let v = [
                    if v[0].is_finite() { v[0] } else { 0.0 },
                    if v[1].is_finite() { v[1] } else { 0.0 },
                    if v[2].is_finite() { v[2] } else { 0.0 },
                ];
                serde_json::json!(v)
            }
            JsonValue::Vec4(v) => {
                let v = [
                    if v[0].is_finite() { v[0] } else { 0.0 },
                    if v[1].is_finite() { v[1] } else { 0.0 },
                    if v[2].is_finite() { v[2] } else { 0.0 },
                    if v[3].is_finite() { v[3] } else { 0.0 },
                ];
                serde_json::json!(v)
            }
            JsonValue::Quat(v) => {
                let v = [
                    if v[0].is_finite() { v[0] } else { 0.0 },
                    if v[1].is_finite() { v[1] } else { 0.0 },
                    if v[2].is_finite() { v[2] } else { 0.0 },
                    if v[3].is_finite() { v[3] } else { 0.0 },
                ];
                serde_json::json!(v)
            }
            JsonValue::Mat2(v) => {
                let v: Vec<f64> = v
                    .iter()
                    .map(|&x| if x.is_finite() { x } else { 0.0 })
                    .collect();
                serde_json::json!(v)
            }
            JsonValue::Mat3(v) => {
                let v: Vec<f64> = v
                    .iter()
                    .map(|&x| if x.is_finite() { x } else { 0.0 })
                    .collect();
                serde_json::json!(v)
            }
            JsonValue::Mat4(v) => {
                let v: Vec<f64> = v
                    .iter()
                    .map(|&x| if x.is_finite() { x } else { 0.0 })
                    .collect();
                serde_json::json!(v)
            }
            JsonValue::Struct(fields) => {
                let mut map = serde_json::Map::new();
                for (key, val) in fields {
                    map.insert(key.clone(), val.to_json_value());
                }
                serde_json::Value::Object(map)
            }
        }
    }
}

pub fn ipc_response_to_json(response: &IpcResponse) -> JsonResponse {
    let payload = match &response.payload {
        Some(IpcResponsePayload::WorldInfo(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::Status(p)) => {
            let mut map = serde_json::Map::new();
            map.insert("tick".to_string(), serde_json::json!(p.tick));
            map.insert("era".to_string(), serde_json::json!(p.era));
            map.insert(
                "sim_time".to_string(),
                serde_json::json!(if p.sim_time.is_finite() {
                    p.sim_time
                } else {
                    0.0
                }),
            );
            map.insert(
                "dt".to_string(),
                serde_json::json!(if p.dt.is_finite() { p.dt } else { 0.0 }),
            );
            map.insert("phase".to_string(), serde_json::json!(p.phase));
            map.insert("running".to_string(), serde_json::json!(p.running));
            Some(serde_json::Value::Object(map))
        }
        Some(IpcResponsePayload::SignalList(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::SignalDescribe(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::FieldList(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::FieldDescribe(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::FieldHistory(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::FieldQuery(p)) => {
            let mut map = serde_json::Map::new();
            map.insert("field_id".to_string(), serde_json::json!(p.field_id));
            map.insert("value".to_string(), p.value.to_json_value());
            Some(serde_json::Value::Object(map))
        }
        Some(IpcResponsePayload::FieldQueryBatch(p)) => {
            let values: Vec<_> = p
                .values
                .iter()
                .map(|v| if v.is_finite() { *v } else { 0.0 })
                .collect();
            let mut map = serde_json::Map::new();
            map.insert("field_id".to_string(), serde_json::json!(p.field_id));
            map.insert("values".to_string(), serde_json::json!(values));
            Some(serde_json::Value::Object(map))
        }
        Some(IpcResponsePayload::FieldLatest(p)) => {
            let mut map = serde_json::Map::new();
            map.insert("field_id".to_string(), serde_json::json!(p.field_id));
            map.insert("tick".to_string(), serde_json::json!(p.tick));
            map.insert(
                "value".to_string(),
                p.value
                    .as_ref()
                    .map(|v| v.to_json_value())
                    .unwrap_or(serde_json::Value::Null),
            );
            Some(serde_json::Value::Object(map))
        }
        Some(IpcResponsePayload::ChroniclePoll(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::ImpulseList(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::ImpulseEmit(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::Playback(p)) => {
            let mut map = serde_json::Map::new();
            map.insert(
                "time".to_string(),
                serde_json::json!(if p.time.is_finite() { p.time } else { 0.0 }),
            );
            Some(serde_json::Value::Object(map))
        }
        Some(IpcResponsePayload::StratumList(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::StratumDescribe(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::EraList(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::EraDescribe(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::EntityList(p)) => serde_json::to_value(p).ok(),
        Some(IpcResponsePayload::EntityDescribe(p)) => serde_json::to_value(p).ok(),
        None => None,
    };
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
        "world.info" => IpcCommand::WorldInfo,
        "signal.list" => IpcCommand::SignalList,
        "signal.describe" => {
            let req: SignalDescribeRequest = serde_json::from_value(request.payload)?;
            IpcCommand::SignalDescribe {
                signal_id: req.signal_id,
            }
        }
        "field.list" => IpcCommand::FieldList,
        "field.describe" => {
            let req: FieldDescribeRequest = serde_json::from_value(request.payload)?;
            IpcCommand::FieldDescribe {
                field_id: req.field_id,
            }
        }
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
        "stratum.list" => IpcCommand::StratumList,
        "stratum.describe" => {
            let req: StratumDescribeRequest = serde_json::from_value(request.payload)?;
            IpcCommand::StratumDescribe {
                stratum_id: req.stratum_id,
            }
        }
        "era.list" => IpcCommand::EraList,
        "era.describe" => {
            let req: EraDescribeRequest = serde_json::from_value(request.payload)?;
            IpcCommand::EraDescribe { era_id: req.era_id }
        }
        "entity.list" => IpcCommand::EntityList,
        "entity.describe" => {
            let req: EntityDescribeRequest = serde_json::from_value(request.payload)?;
            IpcCommand::EntityDescribe {
                entity_id: req.entity_id,
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
        IpcEvent::Tick(p) => {
            let mut map = serde_json::Map::new();
            map.insert("tick".to_string(), serde_json::json!(p.tick));
            map.insert("era".to_string(), serde_json::json!(p.era));
            map.insert(
                "sim_time".to_string(),
                serde_json::json!(if p.sim_time.is_finite() {
                    p.sim_time
                } else {
                    0.0
                }),
            );
            map.insert("field_count".to_string(), serde_json::json!(p.field_count));
            map.insert("event_count".to_string(), serde_json::json!(p.event_count));
            JsonEvent {
                kind: "tick".to_string(),
                payload: serde_json::Value::Object(map),
            }
        }
        IpcEvent::Chronicle(p) => {
            let mut fields = serde_json::Map::new();
            for (k, v) in &p.fields {
                fields.insert(k.clone(), v.to_json_value());
            }
            let mut map = serde_json::Map::new();
            map.insert(
                "chronicle_id".to_string(),
                serde_json::json!(p.chronicle_id),
            );
            map.insert("name".to_string(), serde_json::json!(p.name));
            map.insert("fields".to_string(), serde_json::Value::Object(fields));
            map.insert("tick".to_string(), serde_json::json!(p.tick));
            map.insert("era".to_string(), serde_json::json!(p.era));
            map.insert(
                "sim_time".to_string(),
                serde_json::json!(if p.sim_time.is_finite() {
                    p.sim_time
                } else {
                    0.0
                }),
            );
            JsonEvent {
                kind: "chronicle.event".to_string(),
                payload: serde_json::Value::Object(map),
            }
        }
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
    let value: anyhow::Result<T> = bincode::deserialize(&buffer).map_err(|e| anyhow::anyhow!(e));
    if let Err(ref e) = value {
        tracing::error!("Failed to decode frame of {} bytes: {}", buffer.len(), e);
        tracing::error!("Buffer content: {:02x?}", buffer);
    }
    value.context("decode frame")
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
