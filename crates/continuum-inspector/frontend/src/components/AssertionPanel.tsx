import { useEffect, useState } from 'preact/hooks';
import type { AssertionFailure } from '../types/ipc';

interface AssertionPanelProps {
  ws: any;
  onSelectAssertion: (assertion: AssertionFailure) => void;
}

export function AssertionPanel({ ws, onSelectAssertion }: AssertionPanelProps) {
  const [assertions, setAssertions] = useState<AssertionFailure[]>([]);
  const [filter, setFilter] = useState<'all' | 'warn' | 'error' | 'fatal'>('all');

  useEffect(() => {
    if (ws.status !== 'connected') return;

    const loadAssertions = () => {
      ws.sendRequest('assertion.failures').then((data: any) => {
        setAssertions(data.failures || []);
      }).catch(console.error);
    };

    loadAssertions();

    // Subscribe to assertion events
    const unsubscribe = ws.subscribe('assertion', (msg: any) => {
      if ('type' in msg && msg.type === 'assertion') {
        setAssertions(prev => {
          const updated = [msg.payload, ...prev];
          return updated.slice(0, 200); // Keep last 200
        });
      }
    });

    return unsubscribe;
  }, [ws.status]);

  const filteredAssertions = filter === 'all' 
    ? assertions 
    : assertions.filter(a => a.severity === filter);

  const counts = {
    warn: assertions.filter(a => a.severity === 'warn').length,
    error: assertions.filter(a => a.severity === 'error').length,
    fatal: assertions.filter(a => a.severity === 'fatal').length,
  };

  return (
    <div class="assertion-panel">
      <div class="assertion-header">
        <h3>Assertions</h3>
        <div class="assertion-filters">
          <button 
            class={`filter-btn ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All ({assertions.length})
          </button>
          <button 
            class={`filter-btn severity-warn ${filter === 'warn' ? 'active' : ''}`}
            onClick={() => setFilter('warn')}
          >
            Warn ({counts.warn})
          </button>
          <button 
            class={`filter-btn severity-error ${filter === 'error' ? 'active' : ''}`}
            onClick={() => setFilter('error')}
          >
            Error ({counts.error})
          </button>
          <button 
            class={`filter-btn severity-fatal ${filter === 'fatal' ? 'active' : ''}`}
            onClick={() => setFilter('fatal')}
          >
            Fatal ({counts.fatal})
          </button>
        </div>
      </div>
      <div class="assertion-content">
        {filteredAssertions.length === 0 ? (
          <div class="empty">
            {filter === 'all' 
              ? 'No assertion failures' 
              : `No ${filter} assertions`}
          </div>
        ) : (
          <div class="assertion-list">
            {filteredAssertions.map((assertion, idx) => (
              <div 
                key={idx} 
                class={`assertion-item severity-${assertion.severity}`}
                onClick={() => onSelectAssertion(assertion)}
              >
                <div class="assertion-row">
                  <span class={`assertion-badge severity-${assertion.severity}`}>
                    {assertion.severity.toUpperCase()}
                  </span>
                  <span class="assertion-signal">{assertion.signal_id}</span>
                  <span class="assertion-tick">Tick {assertion.tick}</span>
                </div>
                <div class="assertion-message">{assertion.message}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
