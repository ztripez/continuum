import { useState } from 'preact/hooks';

interface SidebarProps {
  ws: any;
}

export function Sidebar({ ws }: SidebarProps) {
  const [isRunning, setIsRunning] = useState(false);

  const handleStatus = () => {
    ws.sendRequest('status').catch(console.error);
  };

  const handleStep = () => {
    ws.sendRequest('step', { count: 1 }).catch(console.error);
  };

  const handleRun = () => {
    ws.sendRequest('run').then(() => {
      setIsRunning(true);
    }).catch(console.error);
  };

  const handleStop = () => {
    ws.sendRequest('stop').then(() => {
      setIsRunning(false);
    }).catch(console.error);
  };

  return (
    <aside class="sidebar">
      <div class="panel">
        <div class="panel-header">
          <h2>Simulation</h2>
        </div>
        <div class="controls">
          <button onClick={handleStatus}>Status</button>
          <button onClick={handleStep} disabled={isRunning}>Step</button>
          <button onClick={handleRun} disabled={isRunning} class="primary">Run</button>
          <button onClick={handleStop} disabled={!isRunning} class="danger">Stop</button>
        </div>
      </div>
    </aside>
  );
}
