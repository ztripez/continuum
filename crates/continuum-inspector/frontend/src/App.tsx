import { useEffect, useState } from 'preact/hooks';

export function App() {
  const [wsStatus, setWsStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');

  useEffect(() => {
    const ws = new WebSocket(`ws://${location.host}/ws`);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setWsStatus('connected');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWsStatus('disconnected');
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      setWsStatus('disconnected');
    };

    ws.onmessage = (event) => {
      console.log('Message from server:', event.data);
    };

    return () => ws.close();
  }, []);

  return (
    <div class="app">
      <header class="header">
        <h1>Continuum Inspector</h1>
        <div class="status">
          <span class={`status-indicator status-${wsStatus}`}></span>
          <span>{wsStatus === 'connected' ? 'Connected' : wsStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}</span>
        </div>
      </header>
      <main class="main">
        <div class="placeholder">
          <p>Continuum Inspector</p>
          <p class="version">TypeScript + Preact + Vite</p>
        </div>
      </main>
    </div>
  );
}
