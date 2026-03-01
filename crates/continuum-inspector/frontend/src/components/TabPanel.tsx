import { useEffect, useState } from 'preact/hooks';
import type { SignalInfo, FieldInfo, EntityInfo, ChronicleEvent, TreeNode } from '../types/ipc';
import { TreeView } from './TreeView';

export type TabId = 'tree' | 'chronicles' | 'impulses';

interface TabPanelProps {
  currentTab: TabId;
  onTabChange: (tab: TabId) => void;
  onSelectItem: (item: any) => void;
  onEmitImpulse: () => void;
  ws: any;
}

export function TabPanel({ currentTab, onTabChange, onSelectItem, onEmitImpulse, ws }: TabPanelProps) {
  const [tree, setTree] = useState<TreeNode | null>(null);
  const [chronicles, setChronicles] = useState<ChronicleEvent[]>([]);
  const [impulses, setImpulses] = useState<any[]>([]);

  useEffect(() => {
    if (ws.status !== 'connected') return;

    // Load world tree
    ws.sendRequest('world.tree').then((data: TreeNode) => {
      if (data && data.kind) {
        setTree(data);
      }
    }).catch(console.error);

    // Load impulses
    ws.sendRequest('impulse.list').then((data: any) => {
      setImpulses(data || []);
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

  /** Map a tree leaf click to the detail panel selection format. */
  const handleTreeSelect = (node: TreeNode) => {
    const kind = node.kind;
    if (kind === 'signal') {
      ws.sendRequest('signal.describe', { signal_id: node.id }).then((data: SignalInfo) => {
        onSelectItem({ type: 'signal', data });
      }).catch(console.error);
    } else if (kind === 'field') {
      ws.sendRequest('field.describe', { field_id: node.id }).then((data: FieldInfo) => {
        onSelectItem({ type: 'field', data });
      }).catch(console.error);
    } else if (kind === 'entity') {
      ws.sendRequest('entity.describe', { entity_id: node.id }).then((data: EntityInfo) => {
        onSelectItem({ type: 'entity', data });
      }).catch(console.error);
    } else if (kind === 'operator' || kind === 'fracture' || kind === 'chronicle' || kind === 'impulse') {
      // For roles without a dedicated describe handler, show basic info
      onSelectItem({
        type: kind,
        data: { id: node.id, value_type: node.value_type, stratum: node.stratum },
      });
    }
  };

  const handleSelectChronicle = (chronicle: ChronicleEvent) => {
    onSelectItem({ type: 'chronicle', data: chronicle });
  };

  // Count tree leaves for badge
  const countLeaves = (n: TreeNode | null): number => {
    if (!n) return 0;
    if (n.children.length === 0) return 1;
    return n.children.reduce((sum, c) => sum + countLeaves(c), 0);
  };

  return (
    <div class="tab-panel">
      <div class="tabs">
        <button class={currentTab === 'tree' ? 'active' : ''} onClick={() => onTabChange('tree')}>
          Tree ({countLeaves(tree)})
        </button>
        <button class={currentTab === 'chronicles' ? 'active' : ''} onClick={() => onTabChange('chronicles')}>
          Chronicles ({chronicles.length})
        </button>
        <button class={currentTab === 'impulses' ? 'active' : ''} onClick={() => onTabChange('impulses')}>
          Impulses ({impulses.length})
        </button>
      </div>
      <div class="tab-content">
        {currentTab === 'tree' && (
          <TreeView tree={tree} onSelect={handleTreeSelect} />
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
                    Tick {chron.tick ?? '?'} &bull; Era: {chron.era ?? '?'} &bull; {chron.sim_time != null ? chron.sim_time.toFixed(2) + 's' : ''}
                  </div>
                </div>
              ))
            )}
          </div>
        )}
        {currentTab === 'impulses' && (
          <div class="item-list">
            <button class="primary emit-btn" onClick={onEmitImpulse}>
              Emit Impulse...
            </button>
            {impulses.map(imp => (
              <div key={imp.path} class="item" onClick={() => onSelectItem({ type: 'impulse', data: imp })}>
                {imp.path}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
