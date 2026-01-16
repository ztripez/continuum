import { useEffect, useState } from 'preact/hooks';
import { useWebSocket } from './hooks/useWebSocket';
import { Header } from './components/Header';
import { TabPanel } from './components/TabPanel';
import { DetailPanel } from './components/DetailPanel';
import { LogPanel } from './components/LogPanel';
import type { TickEvent } from './types/ipc';

export function App() {
  const ws = useWebSocket(`ws://${location.host}/ws`);
  const [currentTab, setCurrentTab] = useState<'signals' | 'fields' | 'entities' | 'chronicles' | 'assertions'>('signals');
  const [selectedItem, setSelectedItem] = useState<any>(null);
  const [tickInfo, setTickInfo] = useState<TickEvent | null>(null);

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

  return (
    <div class="app">
      <Header status={ws.status} tickInfo={tickInfo} ws={ws} />
      <div class="main-layout">
        <TabPanel 
          currentTab={currentTab}
          onTabChange={setCurrentTab}
          onSelectItem={setSelectedItem}
          ws={ws}
        />
        <div class="right-panel">
          <DetailPanel selectedItem={selectedItem} ws={ws} />
          <LogPanel ws={ws} />
        </div>
      </div>
    </div>
  );
}
