import { useState } from 'preact/hooks';
import type { ConnectionStatus } from '../hooks/useWebSocket';
import type { TickEvent } from '../types/ipc';

interface HeaderProps {
  status: ConnectionStatus;
  tickInfo: TickEvent | null;
  ws: any;
}

export function Header({ status, tickInfo, ws }: HeaderProps) {
  const [isRunning, setIsRunning] = useState(false);

  const handleStep = () => {
    ws.sendRequest('step', { count: 1 }).catch(console.error);
  };

  const handleRun = () => {
    ws.sendRequest('run').then(() => setIsRunning(true)).catch(console.error);
  };

  const handleStop = () => {
    ws.sendRequest('stop').then(() => setIsRunning(false)).catch(console.error);
  };

  return (
    <header class="header">
      <h1>Continuum Inspector</h1>
      
      <div class="header-controls">
        <button onClick={handleStep} disabled={isRunning}>Step</button>
        <button onClick={handleRun} disabled={isRunning} class="primary">Run</button>
        <button onClick={handleStop} disabled={!isRunning} class="danger">Stop</button>
      </div>

      <div class="header-info">
        {tickInfo && (
          <div class="tick-info">
            <span>Tick: {tickInfo.tick}</span>
            <span>Era: {tickInfo.era}</span>
            <span>Phase: {tickInfo.phase}</span>
          </div>
        )}
        <div class="status">
          <span class={`status-indicator status-${status}`}></span>
          <span>{status === 'connected' ? 'Connected' : status === 'connecting' ? 'Connecting...' : 'Disconnected'}</span>
        </div>
      </div>
    </header>
  );
}
