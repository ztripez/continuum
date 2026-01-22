import { useState, useEffect } from 'preact/hooks';
import type { ImpulseInfo } from '../types/ipc';

interface ImpulseModalProps {
  isOpen: boolean;
  onClose: () => void;
  ws: any;
}

export function ImpulseModal({ isOpen, onClose, ws }: ImpulseModalProps) {
  const [impulses, setImpulses] = useState<ImpulseInfo[]>([]);
  const [selectedImpulse, setSelectedImpulse] = useState<ImpulseInfo | null>(null);
  const [payload, setPayload] = useState<string>('');
  const [status, setStatus] = useState<{ type: 'success' | 'error', message: string } | null>(null);

  useEffect(() => {
    if (isOpen && ws.status === 'connected') {
      ws.sendRequest('impulse.list').then((data: any) => {
        setImpulses(data || []);
      }).catch(console.error);
    }
  }, [isOpen, ws.status]);

  const handleEmit = async () => {
    if (!selectedImpulse) return;

    let parsedPayload: any;
    try {
      if (selectedImpulse.payload_type?.toLowerCase().includes('scalar')) {
        parsedPayload = parseFloat(payload);
        if (isNaN(parsedPayload)) throw new Error('Invalid scalar value');
      } else if (selectedImpulse.payload_type?.toLowerCase().includes('bool')) {
        parsedPayload = payload.toLowerCase() === 'true';
      } else if (payload.trim().startsWith('{') || payload.trim().startsWith('[')) {
        parsedPayload = JSON.parse(payload);
      } else {
        parsedPayload = payload;
      }

      const response = await ws.sendRequest('impulse.emit', {
        path: selectedImpulse.path,
        payload: parsedPayload
      });

      setStatus({ type: 'success', message: 'Impulse emitted successfully' });
      setTimeout(() => {
        setStatus(null);
        onClose();
      }, 1500);
    } catch (e: any) {
      setStatus({ type: 'error', message: e.message });
    }
  };

  if (!isOpen) return null;

  return (
    <div class="modal-overlay" onClick={onClose}>
      <div class="modal-content" onClick={e => e.stopPropagation()}>
        <div class="modal-header">
          <h2>Emit Impulse</h2>
          <button class="close-btn" onClick={onClose}>&times;</button>
        </div>
        
        <div class="modal-body">
          <div class="form-group">
            <label>Select Impulse</label>
            <select 
              value={selectedImpulse?.path || ''} 
              onChange={e => {
                const imp = impulses.find(i => i.path === (e.target as HTMLSelectElement).value);
                setSelectedImpulse(imp || null);
                setStatus(null);
              }}
            >
              <option value="">-- Choose an impulse --</option>
              {impulses.map(imp => (
                <option key={imp.path} value={imp.path}>{imp.path}</option>
              ))}
            </select>
          </div>

          {selectedImpulse && (
            <div class="impulse-details">
              {selectedImpulse.doc && <p class="doc-text">{selectedImpulse.doc}</p>}
              <div class="form-group">
                <label>Payload ({selectedImpulse.payload_type || 'Unknown Type'})</label>
                {selectedImpulse.payload_type?.toLowerCase().includes('scalar') ? (
                  <input 
                    type="number" 
                    step="any" 
                    value={payload} 
                    onInput={e => setPayload((e.target as HTMLInputElement).value)} 
                    placeholder="Enter numeric value"
                  />
                ) : selectedImpulse.payload_type?.toLowerCase().includes('bool') ? (
                  <select value={payload} onChange={e => setPayload((e.target as HTMLSelectElement).value)}>
                    <option value="true">True</option>
                    <option value="false">False</option>
                  </select>
                ) : (
                  <textarea 
                    value={payload} 
                    onInput={e => setPayload((e.target as HTMLTextAreaElement).value)} 
                    placeholder='Enter JSON or string payload'
                  />
                )}
              </div>
            </div>
          )}

          {status && (
            <div class={`status-message ${status.type}`}>
              {status.message}
            </div>
          )}
        </div>

        <div class="modal-footer">
          <button class="secondary" onClick={onClose}>Cancel</button>
          <button 
            class="primary" 
            onClick={handleEmit} 
            disabled={!selectedImpulse || !payload}
          >
            Emit
          </button>
        </div>
      </div>
    </div>
  );
}
