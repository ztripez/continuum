import { useEffect, useRef, useState } from 'preact/hooks';

interface LogEntry {
  timestamp: Date;
  type: 'info' | 'event' | 'error';
  message: string;
}

interface LogPanelProps {
  ws: any;
}

/** Server-push event types worth logging. Request/response traffic is filtered out. */
const LOGGABLE_EVENT_TYPES = new Set([
  'assertion_failure',
  'chronicle',
  'era_transition',
  'fault',
  'error',
]);

export function LogPanel({ ws }: LogPanelProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [expanded, setExpanded] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ws.status !== 'connected') return;

    const unsubscribe = ws.subscribe('*', (msg: any) => {
      if ('id' in msg) return;
      if ('type' in msg && msg.type === 'tick') return;

      const msgType = 'type' in msg ? msg.type : '';
      if (!LOGGABLE_EVENT_TYPES.has(msgType)) return;

      const entry: LogEntry = {
        timestamp: new Date(),
        type: msgType === 'error' || msgType === 'fault' || msgType === 'assertion_failure'
          ? 'error'
          : 'event',
        message: JSON.stringify(msg, null, 2),
      };

      setLogs(prev => {
        const updated = [...prev, entry];
        return updated.slice(-200);
      });

      // Auto-expand on new events
      setExpanded(true);
    });

    return unsubscribe;
  }, [ws.status]);

  useEffect(() => {
    if (expanded) {
      logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, expanded]);

  const handleClear = (e: Event) => {
    e.stopPropagation();
    setLogs([]);
  };

  const eventCount = logs.length;

  return (
    <div class={`log-panel ${expanded ? 'expanded' : 'collapsed'}`}>
      <div class="log-header" onClick={() => setExpanded(!expanded)}>
        <h3>
          <span class="log-toggle">{'\u25B2'}</span>
          Log {eventCount > 0 && `(${eventCount})`}
        </h3>
        {eventCount > 0 && (
          <button onClick={handleClear} class="small">Clear</button>
        )}
      </div>
      {expanded && (
        <div class="log-content">
          {logs.length === 0 && (
            <div class="log-entry log-info">
              <span class="log-message">Listening for events...</span>
            </div>
          )}
          {logs.map((entry, idx) => (
            <div key={idx} class={`log-entry log-${entry.type}`}>
              <span class="log-time">[{entry.timestamp.toLocaleTimeString()}]</span>
              <span class="log-message">{entry.message}</span>
            </div>
          ))}
          <div ref={logEndRef} />
        </div>
      )}
    </div>
  );
}
