import { useEffect, useRef, useState } from 'preact/hooks';

interface LogEntry {
  timestamp: Date;
  type: 'info' | 'event' | 'error';
  message: string;
}

interface LogPanelProps {
  ws: any;
}

export function LogPanel({ ws }: LogPanelProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ws.status !== 'connected') return;

    // Subscribe to all messages
    const unsubscribe = ws.subscribe('*', (msg: any) => {
      const entry: LogEntry = {
        timestamp: new Date(),
        type: 'type' in msg && msg.type.includes('event') ? 'event' : 'error' in msg ? 'error' : 'info',
        message: JSON.stringify(msg, null, 2),
      };
      
      setLogs(prev => {
        const updated = [...prev, entry];
        return updated.slice(-500); // Keep last 500
      });
    });

    return unsubscribe;
  }, [ws.status]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handleClear = () => {
    setLogs([]);
  };

  return (
    <div class="log-panel">
      <div class="log-header">
        <h3>Log</h3>
        <button onClick={handleClear} class="small">Clear</button>
      </div>
      <div class="log-content">
        {logs.map((entry, idx) => (
          <div key={idx} class={`log-entry log-${entry.type}`}>
            <span class="log-time">[{entry.timestamp.toLocaleTimeString()}]</span>
            <span class="log-message">{entry.message}</span>
          </div>
        ))}
        <div ref={logEndRef} />
      </div>
    </div>
  );
}
