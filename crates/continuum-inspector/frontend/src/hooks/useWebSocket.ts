import { useEffect, useRef, useState } from 'preact/hooks';
import type { IpcMessage, JsonRequest } from '../types/ipc';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';

export function useWebSocket(url: string) {
  const [status, setStatus] = useState<ConnectionStatus>('connecting');
  const wsRef = useRef<WebSocket | null>(null);
  const requestIdRef = useRef(1);
  const listenersRef = useRef<Map<string, Set<(msg: IpcMessage) => void>>>(new Map());

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setStatus('connected');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setStatus('disconnected');
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      setStatus('disconnected');
    };

    ws.onmessage = (event) => {
      try {
        const msg: IpcMessage = JSON.parse(event.data);
        
        // Notify type-specific listeners
        const type = 'type' in msg ? msg.type : 'response';
        const listeners = listenersRef.current.get(type);
        if (listeners) {
          listeners.forEach(cb => cb(msg));
        }

        // Notify wildcard listeners
        const wildcardListeners = listenersRef.current.get('*');
        if (wildcardListeners) {
          wildcardListeners.forEach(cb => cb(msg));
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    return () => {
      ws.close();
    };
  }, [url]);

  const sendRequest = (type: string, payload?: any): Promise<any> => {
    return new Promise((resolve, reject) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket not connected'));
        return;
      }

      const id = requestIdRef.current++;
      const request: JsonRequest = { id, type, payload };

      // Listen for response with this ID
      const handleResponse = (msg: IpcMessage) => {
        if ('id' in msg && msg.id === id) {
          unsubscribe();
          if (msg.ok) {
            resolve(msg.payload);
          } else {
            reject(new Error(msg.error || 'Request failed'));
          }
        }
      };

      const unsubscribe = subscribe('response', handleResponse);

      try {
        wsRef.current.send(JSON.stringify(request));
      } catch (err) {
        unsubscribe();
        reject(err);
      }

      // Timeout after 30 seconds
      setTimeout(() => {
        unsubscribe();
        reject(new Error('Request timeout'));
      }, 30000);
    });
  };

  const subscribe = (type: string, callback: (msg: IpcMessage) => void): (() => void) => {
    if (!listenersRef.current.has(type)) {
      listenersRef.current.set(type, new Set());
    }
    listenersRef.current.get(type)!.add(callback);

    return () => {
      const listeners = listenersRef.current.get(type);
      if (listeners) {
        listeners.delete(callback);
        if (listeners.size === 0) {
          listenersRef.current.delete(type);
        }
      }
    };
  };

  return {
    status,
    sendRequest,
    subscribe,
  };
}
