import { useEffect, useState } from 'preact/hooks';
import type { SignalInfo, FieldInfo, EntityInfo, ChronicleEvent, AssertionFailure } from '../types/ipc';

interface TabPanelProps {
  currentTab: 'signals' | 'fields' | 'entities' | 'chronicles' | 'assertions';
  onTabChange: (tab: 'signals' | 'fields' | 'entities' | 'chronicles' | 'assertions') => void;
  onSelectItem: (item: any) => void;
  ws: any;
}

export function TabPanel({ currentTab, onTabChange, onSelectItem, ws }: TabPanelProps) {
  const [signals, setSignals] = useState<string[]>([]);
  const [fields, setFields] = useState<string[]>([]);
  const [entities, setEntities] = useState<string[]>([]);
  const [chronicles, setChronicles] = useState<ChronicleEvent[]>([]);
  const [assertions, setAssertions] = useState<AssertionFailure[]>([]);

  useEffect(() => {
    if (ws.status !== 'connected') return;

    // Load lists
    ws.sendRequest('signal.list').then((data: any) => {
      setSignals(data.signals || []);
    }).catch(console.error);

    ws.sendRequest('field.list').then((data: any) => {
      setFields(data.fields || []);
    }).catch(console.error);

    ws.sendRequest('entity.list').then((data: any) => {
      setEntities(data.entities || []);
    }).catch(console.error);

    ws.sendRequest('assertion.failures').then((data: any) => {
      setAssertions(data.failures || []);
    }).catch(console.error);

    // Subscribe to chronicle events
    const unsubscribe = ws.subscribe('chronicle.event', (msg: any) => {
      if ('type' in msg && msg.type === 'chronicle.event') {
        setChronicles(prev => {
          const updated = [...prev, { ...msg.payload, timestamp: new Date() }];
          return updated.slice(-200); // Keep last 200
        });
      }
    });

    return unsubscribe;
  }, [ws.status]);

  const handleSelectSignal = (id: string) => {
    ws.sendRequest('signal.describe', { signal_id: id }).then((data: SignalInfo) => {
      onSelectItem({ type: 'signal', data });
    }).catch(console.error);
  };

  const handleSelectField = (id: string) => {
    ws.sendRequest('field.describe', { field_id: id }).then((data: FieldInfo) => {
      onSelectItem({ type: 'field', data });
    }).catch(console.error);
  };

  const handleSelectEntity = (id: string) => {
    ws.sendRequest('entity.describe', { entity_id: id }).then((data: EntityInfo) => {
      onSelectItem({ type: 'entity', data });
    }).catch(console.error);
  };

  const handleSelectChronicle = (chronicle: ChronicleEvent) => {
    onSelectItem({ type: 'chronicle', data: chronicle });
  };

  return (
    <div class="tab-panel">
      <div class="tabs">
        <button class={currentTab === 'signals' ? 'active' : ''} onClick={() => onTabChange('signals')}>
          Signals ({signals.length})
        </button>
        <button class={currentTab === 'fields' ? 'active' : ''} onClick={() => onTabChange('fields')}>
          Fields ({fields.length})
        </button>
        <button class={currentTab === 'entities' ? 'active' : ''} onClick={() => onTabChange('entities')}>
          Entities ({entities.length})
        </button>
        <button class={currentTab === 'chronicles' ? 'active' : ''} onClick={() => onTabChange('chronicles')}>
          Chronicles ({chronicles.length})
        </button>
        <button class={currentTab === 'assertions' ? 'active' : ''} onClick={() => onTabChange('assertions')}>
          Assertions ({assertions.length})
        </button>
      </div>
      <div class="tab-content">
        {currentTab === 'signals' && (
          <div class="item-list">
            {signals.map(id => (
              <div key={id} class="item" onClick={() => handleSelectSignal(id)}>
                {id}
              </div>
            ))}
          </div>
        )}
        {currentTab === 'fields' && (
          <div class="item-list">
            {fields.map(id => (
              <div key={id} class="item" onClick={() => handleSelectField(id)}>
                {id}
              </div>
            ))}
          </div>
        )}
        {currentTab === 'entities' && (
          <div class="item-list">
            {entities.map(id => (
              <div key={id} class="item" onClick={() => handleSelectEntity(id)}>
                {id}
              </div>
            ))}
          </div>
        )}
        {currentTab === 'chronicles' && (
          <div class="item-list">
            {chronicles.length === 0 ? (
              <div class="empty">No chronicle events yet...</div>
            ) : (
              [...chronicles].reverse().map((chron, idx) => (
                <div key={idx} class="item chronicle-item" onClick={() => handleSelectChronicle(chron)}>
                  <div class="chronicle-name">{chron.name}</div>
                  <div class="chronicle-meta">
                    Tick {chron.tick} • Era: {chron.era} • {chron.sim_time.toFixed(2)}s
                  </div>
                </div>
              ))
            )}
          </div>
        )}
        {currentTab === 'assertions' && (
          <div class="item-list">
            {assertions.length === 0 ? (
              <div class="empty">No assertion failures</div>
            ) : (
              [...assertions].reverse().map((assertion, idx) => (
                <div 
                  key={idx} 
                  class={`item assertion-item severity-${assertion.severity}`}
                  onClick={() => onSelectItem({ type: 'assertion', data: assertion })}
                >
                  <div class="assertion-signal">{assertion.signal_id}</div>
                  <div class="assertion-severity">{assertion.severity.toUpperCase()}</div>
                  <div class="assertion-message">{assertion.message}</div>
                  <div class="assertion-meta">
                    Tick {assertion.tick} • Era: {assertion.era} • {assertion.sim_time.toFixed(2)}s
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}
