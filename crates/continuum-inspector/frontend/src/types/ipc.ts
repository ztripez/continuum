// IPC protocol types matching Rust definitions

export interface JsonRequest {
  id: number;
  type: string;
  payload?: any;
}

export interface JsonResponse {
  id: number;
  ok: boolean;
  error?: string;
  payload?: any;
}

export interface JsonEvent {
  type: string;
  payload: any;
}

export type IpcMessage = JsonResponse | JsonEvent;

// Payload types
export interface SignalInfo {
  id: string;
  title?: string;
  symbol?: string;
  doc?: string;
  value_type: string;
  unit?: string;
  range?: [number, number];
  stratum?: string;
}

export interface FieldInfo {
  id: string;
  title?: string;
  symbol?: string;
  doc?: string;
  topology: string;
  value_type: string;
  unit?: string;
  range?: [number, number];
}

export interface EntityInfo {
  id: string;
  title?: string;
  doc?: string;
  count_bounds?: [number, number];
}

export interface StratumInfo {
  id: string;
  title?: string;
  doc?: string;
  default_stride?: number;
}

export interface EraInfo {
  id: string;
  title?: string;
  doc?: string;
  dt_seconds?: number;
  is_initial?: boolean;
  is_terminal?: boolean;
}

export interface WorldInfo {
  strata: StratumInfo[];
  eras: EraInfo[];
}

export interface ImpulseInfo {
  id: string;
  title?: string;
  symbol?: string;
  doc?: string;
  payload_type: string;
  unit?: string;
  range?: [number, number];
}

export interface ChronicleEvent {
  chronicle_id: string;
  name: string;
  fields: [string, any][];
  tick: number;
  era: string;
  sim_time: number;
}

export interface TickEvent {
  tick: number;
  era: string;
  sim_time: number;
  field_count: number;
  event_count: number;
  phase: string;
}
