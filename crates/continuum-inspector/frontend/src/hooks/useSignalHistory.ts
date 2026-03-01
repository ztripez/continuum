import { useEffect, useRef, useState } from 'preact/hooks';
import type { SignalHistoryData } from '../types/ipc';

/** Fetches and caches signal history from the backend. */
export function useSignalHistory(
  sendRequest: (type: string, payload?: any) => Promise<any>,
  signalId: string | null,
  pollIntervalMs = 1000
) {
  const [history, setHistory] = useState<SignalHistoryData | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!signalId) {
      setHistory(null);
      return;
    }

    const fetchHistory = async () => {
      try {
        const data: SignalHistoryData = await sendRequest('signal.history', {
          signal_id: signalId,
        });
        setHistory(data);
      } catch {
        // Silently ignore fetch errors — will retry on next poll
      }
    };

    // Fetch immediately, then poll
    fetchHistory();
    intervalRef.current = setInterval(fetchHistory, pollIntervalMs);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [signalId, pollIntervalMs]);

  return history;
}
