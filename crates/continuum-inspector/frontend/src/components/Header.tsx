import type { ConnectionStatus } from '../hooks/useWebSocket';
import type { TickEvent } from '../types/ipc';

interface HeaderProps {
  status: ConnectionStatus;
  tickInfo: TickEvent | null;
  ws: any;
}

export function Header({ status, tickInfo, ws }: HeaderProps) {
  return (
    <header class="header">
      <h1>Continuum Inspector</h1>
      <div class="header-info">
        {tickInfo && (
          <div class="tick-info">
            <span>Tick: {tickInfo.tick}</span>
            <span>Era: {tickInfo.era}</span>
            <span>Time: {tickInfo.sim_time.toFixed(2)}s</span>
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
