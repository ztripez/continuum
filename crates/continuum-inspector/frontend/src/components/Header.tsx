import { useEffect, useState } from 'preact/hooks';
import { SimulationControl } from './SimulationControl';
import type { ConnectionStatus } from '../hooks/useWebSocket';
import type { TickEvent } from '../types/ipc';

interface HeaderProps {
  status: ConnectionStatus;
  tickInfo: TickEvent | null;
  ws: any;
  hasErrors: boolean;
  onSimulationChange?: () => void;
}

type SimStatus = 'stopped' | 'running' | 'paused' | 'error' | 'warmup';

export function Header({ status, tickInfo, ws, hasErrors, onSimulationChange }: HeaderProps) {
  const [showSimControl, setShowSimControl] = useState(false);
  const [simStatus, setSimStatus] = useState<SimStatus>('stopped');
  const [warmupComplete, setWarmupComplete] = useState<boolean>(true);
  const [lastError, setLastError] = useState<string | null>(null);
  const [tickRate, setTickRate] = useState<number>(60); // Default 60 tps
  const [showTickRateControl, setShowTickRateControl] = useState(false);

  useEffect(() => {
    if (hasErrors) {
      setSimStatus('error');
    }
  }, [hasErrors]);

  // Update status from tick events
  useEffect(() => {
    if (!tickInfo) return;

    // Backend always emits execution_state (stopped/running/paused/error)
    if (tickInfo.execution_state) {
      setSimStatus(tickInfo.execution_state as SimStatus);
    }
    
    // Track warmup separately - warmup is a phase, not a state
    // If running and warmup not complete, show warmup status
    if (tickInfo.warmup_complete !== undefined) {
      setWarmupComplete(tickInfo.warmup_complete);
      if (!tickInfo.warmup_complete && tickInfo.execution_state === 'running') {
        setSimStatus('warmup');
      }
    }
    
    if (tickInfo.last_error) {
      setLastError(tickInfo.last_error);
    }
    
    if (tickInfo.tick_rate !== undefined) {
      setTickRate(tickInfo.tick_rate);
    }
  }, [tickInfo]);

  // Fetch initial status on connect
  useEffect(() => {
    if (ws.status !== 'connected') return;

    ws.sendRequest('status').then((payload: any) => {
      // Backend always emits execution_state
      if (payload.execution_state) {
        setSimStatus(payload.execution_state);
      }
      
      if (payload.warmup_complete !== undefined) {
        setWarmupComplete(payload.warmup_complete);
      }
      
      if (payload.tick_rate !== undefined) {
        setTickRate(payload.tick_rate);
      }
      
      if (payload.last_error) {
        setLastError(payload.last_error);
      }
    }).catch((err: any) => {
      setLastError(err.message || 'Failed to fetch status');
      setSimStatus('error');
    });
  }, [ws.status]);

  const handleStep = () => {
    // If warmup not complete, show warmup status during step
    if (!warmupComplete) {
      setSimStatus('warmup');
    }
    ws.sendRequest('step', { count: 1 })
      .then((payload: any) => {
        setWarmupComplete(payload.warmup_complete ?? true);
        setSimStatus('stopped');
      })
      .catch((err: any) => {
        setLastError(err.message || 'Step failed');
        setSimStatus('error');
      });
  };

  const handleRun = () => {
    // If warmup not complete, show warmup status first
    if (!warmupComplete) {
      setSimStatus('warmup');
    }
    ws.sendRequest('run', { tick_rate: tickRate })
      .then((payload: any) => {
        setWarmupComplete(payload.warmup_complete ?? true);
        setSimStatus('running');
      })
      .catch((err: any) => {
        setLastError(err.message || 'Run failed');
        setSimStatus('error');
      });
  };

  const handlePause = () => {
    ws.sendRequest('pause')
      .then(() => setSimStatus('paused'))
      .catch((err: any) => {
        setLastError(err.message || 'Pause failed');
        setSimStatus('error');
      });
  };

  const handleResume = () => {
    ws.sendRequest('resume')
      .then(() => setSimStatus('running'))
      .catch((err: any) => {
        setLastError(err.message || 'Resume failed');
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
      case 'warmup': return 'WARMUP';
      case 'error': return lastError || 'ERROR';
    }
  };

  return (
    <>
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
            onClick={() => setShowSimControl(!showSimControl)}
            class="control-btn"
            title="Simulation lifecycle controls"
          >
            <span class="icon">⚙</span>
            {showSimControl ? 'Hide' : 'Sim'} Controls
          </button>
          <button 
            onClick={() => setShowTickRateControl(!showTickRateControl)}
            class="control-btn"
            title="Tick rate control"
          >
            <span class="icon">⏱</span>
            {tickRate === 0 ? '∞' : tickRate} tps
          </button>
          <button 
            onClick={handleStep} 
            disabled={simStatus === 'running'}
            class="control-btn"
            title="Step one tick"
          >
            <span class="icon">⏭</span>
            Step
          </button>
          {simStatus === 'running' ? (
            <button 
              onClick={handlePause} 
              class="control-btn"
              title="Pause simulation"
            >
              <span class="icon">⏸</span>
              Pause
            </button>
          ) : simStatus === 'paused' || simStatus === 'error' ? (
            <button 
              onClick={handleResume} 
              class="control-btn primary"
              title="Resume simulation"
            >
              <span class="icon">▶</span>
              Resume
            </button>
          ) : (
            <button 
              onClick={handleRun} 
              disabled={simStatus === 'running'} 
              class="control-btn primary"
              title="Run simulation"
            >
              <span class="icon">▶</span>
              Run
            </button>
          )}
          <button 
            onClick={handleStop} 
            disabled={simStatus === 'stopped'} 
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
      
      {simStatus === 'error' && lastError && (
        <div class="error-banner">
          <span class="error-icon">⚠</span>
          <span class="error-message">Error: {lastError}</span>
          <button 
            onClick={handleResume}
            class="control-btn primary"
            title="Resume from error"
          >
            Resume
          </button>
        </div>
      )}
      
      {showTickRateControl && (
        <div class="tick-rate-panel">
          <label class="tick-rate-label">
            Tick Rate: {tickRate === 0 ? 'Unlimited' : `${tickRate} tps`}
          </label>
          <input 
            type="range" 
            min="0" 
            max="120" 
            step="1"
            value={tickRate}
            onInput={(e) => {
              const val = parseInt((e.target as HTMLInputElement).value);
              setTickRate(val);
            }}
            class="tick-rate-slider"
          />
          <div class="tick-rate-presets">
            <button onClick={() => setTickRate(1)} class="preset-btn">1</button>
            <button onClick={() => setTickRate(10)} class="preset-btn">10</button>
            <button onClick={() => setTickRate(30)} class="preset-btn">30</button>
            <button onClick={() => setTickRate(60)} class="preset-btn">60</button>
            <button onClick={() => setTickRate(120)} class="preset-btn">120</button>
            <button onClick={() => setTickRate(0)} class="preset-btn">∞</button>
          </div>
        </div>
      )}
      
      {showSimControl && (
        <div class="sim-control-panel">
          <SimulationControl onSimulationChange={() => {
            setShowSimControl(false);
            onSimulationChange?.();
          }} />
        </div>
      )}
    </>
  );
}
