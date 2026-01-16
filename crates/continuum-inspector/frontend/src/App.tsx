import { useEffect, useState } from 'preact/hooks';
import { useWebSocket } from './hooks/useWebSocket';
import { Header } from './components/Header';
import { TabPanel } from './components/TabPanel';
import { DetailPanel } from './components/DetailPanel';
import { LogPanel } from './components/LogPanel';
import { AssertionPanel } from './components/AssertionPanel';
import type { TickEvent, AssertionFailure } from './types/ipc';

export function App() {
  const ws = useWebSocket(`ws://${location.host}/ws`);
  const [currentTab, setCurrentTab] = useState<'signals' | 'fields' | 'entities' | 'chronicles'>('signals');
  const [selectedItem, setSelectedItem] = useState<any>(null);
  const [tickInfo, setTickInfo] = useState<TickEvent | null>(null);
  const [assertionCount, setAssertionCount] = useState(0);

  // Subscribe to tick events and fetch initial status
  useEffect(() => {
    if (ws.status !== 'connected') return;

    const unsubscribe = ws.subscribe('tick', (msg) => {
      if ('type' in msg && msg.type === 'tick') {
        setTickInfo(msg.payload as TickEvent);
      }
    });

    // Request initial status
    ws.sendRequest('status').then((payload: any) => {
      if (payload) {
        setTickInfo(payload as TickEvent);
      }
    }).catch(console.error);

    return unsubscribe;
  }, [ws.status]);

  // Track assertion count
  useEffect(() => {
    if (ws.status !== 'connected') return;

    ws.sendRequest('assertion.failures').then((data: any) => {
      setAssertionCount((data.failures || []).length);
    }).catch(console.error);

    const unsubscribe = ws.subscribe('assertion', () => {
      setAssertionCount(prev => prev + 1);
    });

    return unsubscribe;
  }, [ws.status]);

  const handleSelectAssertion = (assertion: AssertionFailure) => {
    setSelectedItem({ type: 'assertion', data: assertion });
  };

  return (
    <div class="app">
      <Header 
        status={ws.status} 
        tickInfo={tickInfo} 
        ws={ws} 
        hasErrors={assertionCount > 0}
      />
      <div class="main-layout">
        <TabPanel 
          currentTab={currentTab}
          onTabChange={setCurrentTab}
          onSelectItem={setSelectedItem}
          ws={ws}
        />
        <div class="center-panel">
          <DetailPanel selectedItem={selectedItem} ws={ws} />
          <LogPanel ws={ws} />
        </div>
        <AssertionPanel ws={ws} onSelectAssertion={handleSelectAssertion} />
      </div>
    </div>
  );
}
