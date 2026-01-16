import { useEffect, useState } from 'preact/hooks';
import type { ConnectionStatus } from '../hooks/useWebSocket';
import type { TickEvent } from '../types/ipc';

interface HeaderProps {
  status: ConnectionStatus;
  tickInfo: TickEvent | null;
  ws: any;
  hasErrors: boolean;
}

type SimStatus = 'stopped' | 'running' | 'paused' | 'error';

export function Header({ status, tickInfo, ws, hasErrors }: HeaderProps) {
  const [simStatus, setSimStatus] = useState<SimStatus>('stopped');
  const [lastError, setLastError] = useState<string | null>(null);

  useEffect(() => {
    if (hasErrors) {
      setSimStatus('error');
    }
  }, [hasErrors]);

  useEffect(() => {
    if (ws.status !== 'connected') return;

    ws.sendRequest('status').then((payload: any) => {
      setSimStatus(payload.running ? 'running' : 'stopped');
    }).catch(console.error);
  }, [ws.status]);

  const handleStep = () => {
    ws.sendRequest('step', { count: 1 })
      .then(() => setSimStatus('stopped'))
      .catch((err: any) => {
        setLastError(err.message || 'Step failed');
        setSimStatus('error');
      });
  };

  const handleRun = () => {
    ws.sendRequest('run')
      .then(() => setSimStatus('running'))
      .catch((err: any) => {
        setLastError(err.message || 'Run failed');
        setSimStatus('error');
      });
  };

  const handleStop = () => {
    ws.sendRequest('stop')
      .then(() => setSimStatus('stopped'))
      .catch(console.error);
  };

  const getSimStatusLabel = () => {
    switch (simStatus) {
      case 'running': return 'RUNNING';
      case 'stopped': return 'STOPPED';
      case 'paused': return 'PAUSED';
      case 'error': return lastError || 'ERROR';
    }
  };

  return (
    <header class="header">
      <div class="header-left">
        <h1>Continuum Inspector</h1>
        <div class={`sim-status sim-status-${simStatus}`}>
          <span class="sim-status-indicator"></span>
          <span class="sim-status-label">{getSimStatusLabel()}</span>
        </div>
      </div>
      
      <div class="header-controls">
        <button 
          onClick={handleStep} 
          disabled={simStatus === 'running'}
          class="control-btn"
          title="Step one tick"
        >
          <span class="icon">⏭</span>
          Step
        </button>
        <button 
          onClick={handleRun} 
          disabled={simStatus === 'running'} 
          class="control-btn primary"
          title="Run simulation"
        >
          <span class="icon">▶</span>
          Run
        </button>
        <button 
          onClick={handleStop} 
          disabled={simStatus === 'stopped' || simStatus === 'error'} 
          class="control-btn danger"
          title="Stop simulation"
        >
          <span class="icon">⏹</span>
          Stop
        </button>
      </div>

      <div class="header-info">
        {tickInfo && (
          <div class="tick-info">
            <div class="info-item">
              <span class="info-label">Tick</span>
              <span class="info-value">{tickInfo.tick}</span>
            </div>
            <div class="info-item">
              <span class="info-label">Era</span>
              <span class="info-value">{tickInfo.era}</span>
            </div>
            <div class="info-item">
              <span class="info-label">Time</span>
              <span class="info-value">{tickInfo.sim_time?.toFixed(2)}s</span>
            </div>
          </div>
        )}
        <div class={`connection-status connection-${status}`}>
          <span class="connection-indicator"></span>
          <span class="connection-label">
            {status === 'connected' ? 'Connected' : status === 'connecting' ? 'Connecting...' : 'Disconnected'}
          </span>
        </div>
      </div>
    </header>
  );
}
