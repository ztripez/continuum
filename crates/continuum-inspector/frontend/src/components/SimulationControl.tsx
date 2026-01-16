import { useState } from 'preact/hooks';

interface SimulationControlProps {
  onSimulationChange?: () => void;
}

export function SimulationControl({ onSimulationChange }: SimulationControlProps) {
  const [worldPath, setWorldPath] = useState('');
  const [scenario, setScenario] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);

  const showMessage = (type: 'success' | 'error', text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 5000);
  };

  const loadSimulation = async () => {
    if (!worldPath.trim()) {
      showMessage('error', 'Please enter a world path');
      return;
    }

    setLoading(true);
    try {
      const payload: { world_path: string; scenario?: string } = { world_path: worldPath };
      if (scenario.trim()) {
        payload.scenario = scenario.trim();
      }

      const response = await fetch('/api/sim/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      
      if (data.success) {
        showMessage('success', data.message);
        onSimulationChange?.();
      } else {
        showMessage('error', data.message);
      }
    } catch (err) {
      showMessage('error', `Failed to load simulation: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const restartSimulation = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/sim/restart', {
        method: 'POST',
      });

      const data = await response.json();
      
      if (data.success) {
        showMessage('success', data.message);
        onSimulationChange?.();
      } else {
        showMessage('error', data.message);
      }
    } catch (err) {
      showMessage('error', `Failed to restart simulation: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const stopSimulation = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/sim/stop', {
        method: 'POST',
      });

      const data = await response.json();
      
      if (data.success) {
        showMessage('success', data.message);
        onSimulationChange?.();
      } else {
        showMessage('error', data.message);
      }
    } catch (err) {
      showMessage('error', `Failed to stop simulation: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div class="simulation-control">
      <div class="control-row">
        <input
          type="text"
          value={worldPath}
          onInput={(e) => setWorldPath((e.target as HTMLInputElement).value)}
          placeholder="Path to world directory (e.g., examples/terra)"
          class="world-path-input"
          disabled={loading}
        />
        <input
          type="text"
          value={scenario}
          onInput={(e) => setScenario((e.target as HTMLInputElement).value)}
          placeholder="Scenario (optional, e.g., default)"
          class="scenario-input"
          disabled={loading}
        />
        <button
          onClick={loadSimulation}
          disabled={loading || !worldPath.trim()}
          class="control-btn primary"
        >
          {loading ? 'Loading...' : 'Load World'}
        </button>
      </div>

      <div class="control-row">
        <button
          onClick={restartSimulation}
          disabled={loading}
          class="control-btn"
        >
          Restart
        </button>
        <button
          onClick={stopSimulation}
          disabled={loading}
          class="control-btn danger"
        >
          Stop Simulation
        </button>
      </div>

      {message && (
        <div class={`message message-${message.type}`}>
          {message.text}
        </div>
      )}
    </div>
  );
}
