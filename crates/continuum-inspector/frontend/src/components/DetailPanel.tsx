import { useEffect, useState } from 'preact/hooks';
import type {
  FieldSampleData,
  FieldTopology,
  SignalHistoryData,
} from '../types/ipc';
import { ScalarLineChart } from './charts/ScalarLineChart';
import { PointCloudScatter } from './charts/PointCloudScatter';
import { GridHeatmap } from './charts/GridHeatmap';
import { GlobeProjection } from './charts/GlobeProjection';
import { useSignalHistory } from '../hooks/useSignalHistory';

interface DetailPanelProps {
  selectedItem: any;
  sendRequest: (type: string, payload?: any) => Promise<any>;
}

/** Renders a field sample chart based on the field's topology. */
function FieldChart({
  fieldId,
  topology,
  sendRequest,
}: {
  fieldId: string;
  topology: FieldTopology | null;
  sendRequest: (type: string, payload?: any) => Promise<any>;
}) {
  const [samples, setSamples] = useState<FieldSampleData | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchSamples = async () => {
      try {
        const data: FieldSampleData = await sendRequest('field.samples', {
          field_id: fieldId,
        });
        if (!cancelled) setSamples(data);
      } catch {
        // Ignore errors, will retry on next poll
      }
    };

    fetchSamples();
    const interval = setInterval(fetchSamples, 1000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [fieldId]);

  if (!samples || samples.samples.length === 0) {
    return <div class="chart-empty">No samples available</div>;
  }

  switch (topology) {
    case 'sphere_surface':
      return (
        <GlobeProjection
          samples={samples.samples}
          label={`${fieldId} (tick ${samples.tick})`}
        />
      );
    case 'grid2d':
      return (
        <GridHeatmap
          samples={samples.samples}
          label={`${fieldId} (tick ${samples.tick})`}
        />
      );
    case 'point_cloud':
    default:
      return (
        <PointCloudScatter
          samples={samples.samples}
          label={`${fieldId} (tick ${samples.tick})`}
        />
      );
  }
}

/** Renders a signal history line chart. */
function SignalChart({
  signalId,
  sendRequest,
}: {
  signalId: string;
  sendRequest: (type: string, payload?: any) => Promise<any>;
}) {
  const history = useSignalHistory(sendRequest, signalId);

  if (!history || history.entries.length === 0) {
    return <div class="chart-empty">No history data</div>;
  }

  // Only show line chart for scalar signals
  const hasScalar = history.entries.some((e) => e.scalar != null);
  if (!hasScalar) {
    return <div class="chart-empty">Non-scalar signal (no chart)</div>;
  }

  return <ScalarLineChart entries={history.entries} label={signalId} />;
}

export function DetailPanel({ selectedItem, sendRequest }: DetailPanelProps) {
  if (!selectedItem) {
    return (
      <div class="detail-panel">
        <div class="empty">Select an item to view details</div>
      </div>
    );
  }

  const { type, data } = selectedItem;

  return (
    <div class="detail-panel">
      <div class="detail-header">
        <h2>{data.title || data.id || data.name}</h2>
        <div class="detail-type">{type}</div>
      </div>
      {data.doc && <div class="detail-doc">{data.doc}</div>}
      <div class="detail-content">
        {type === 'signal' && (
          <>
            <div class="detail-grid">
              <span class="label">ID:</span>
              <span>{data.id}</span>
              <span class="label">Type:</span>
              <span>{data.value_type}</span>
              {data.unit && (
                <>
                  <span class="label">Unit:</span>
                  <span>{data.unit}</span>
                </>
              )}
              {data.stratum && (
                <>
                  <span class="label">Stratum:</span>
                  <span>{data.stratum}</span>
                </>
              )}
              {data.range && (
                <>
                  <span class="label">Range:</span>
                  <span>
                    {data.range[0]} … {data.range[1]}
                  </span>
                </>
              )}
            </div>
            <div class="detail-chart">
              <SignalChart signalId={data.id} sendRequest={sendRequest} />
            </div>
          </>
        )}
        {type === 'field' && (
          <>
            <div class="detail-grid">
              <span class="label">ID:</span>
              <span>{data.id}</span>
              <span class="label">Topology:</span>
              <span>{data.topology || 'unknown'}</span>
              <span class="label">Type:</span>
              <span>{data.value_type}</span>
              {data.unit && (
                <>
                  <span class="label">Unit:</span>
                  <span>{data.unit}</span>
                </>
              )}
              {data.range && (
                <>
                  <span class="label">Range:</span>
                  <span>
                    {data.range[0]} … {data.range[1]}
                  </span>
                </>
              )}
            </div>
            <div class="detail-chart">
              <FieldChart
                fieldId={data.id}
                topology={data.topology as FieldTopology}
                sendRequest={sendRequest}
              />
            </div>
          </>
        )}
        {type === 'entity' && (
          <>
            <div class="detail-grid">
              <span class="label">ID:</span>
              <span>{data.id}</span>
              {data.count_bounds && (
                <>
                  <span class="label">Count Bounds:</span>
                  <span>
                    {data.count_bounds[0]} … {data.count_bounds[1]}
                  </span>
                </>
              )}
            </div>
            {data.members && data.members.length > 0 && (
              <div class="detail-section">
                <h3>Members ({data.members.length})</h3>
                <div class="member-list">
                  {data.members.map((m: any) => (
                    <div key={m.id} class="member-item">
                      <div class="member-name">{m.id}</div>
                      <div class="member-meta">
                        <span class="member-role">{m.role}</span>
                        {m.value_type && <span class="member-type">{m.value_type}</span>}
                        {m.stratum && <span class="member-stratum">{m.stratum}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
        {type === 'chronicle' && (
          <>
            <div class="detail-grid">
              <span class="label">Event:</span>
              <span>{data.name}</span>
              <span class="label">Tick:</span>
              <span>{data.tick}</span>
              <span class="label">Era:</span>
              <span>{data.era}</span>
              <span class="label">Sim Time:</span>
              <span>{data.sim_time != null ? data.sim_time.toFixed(2) + 's' : '—'}</span>
            </div>
            {data.fields && data.fields.length > 0 && (
              <div class="detail-section">
                <h3>Fields</h3>
                <div class="detail-grid">
                  {data.fields.map(([key, value]: [string, any]) => (
                    <>
                      <span class="label">{key}:</span>
                      <span>{JSON.stringify(value)}</span>
                    </>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
        {type === 'assertion' && (
          <>
            <div class="detail-grid">
              <span class="label">Signal:</span>
              <span>{data.signal_id}</span>
              <span class="label">Severity:</span>
              <span class={`severity-${data.severity}`}>
                {data.severity.toUpperCase()}
              </span>
              <span class="label">Message:</span>
              <span>{data.message}</span>
              <span class="label">Tick:</span>
              <span>{data.tick}</span>
              <span class="label">Era:</span>
              <span>{data.era}</span>
              <span class="label">Sim Time:</span>
              <span>{data.sim_time != null ? data.sim_time.toFixed(2) + 's' : '—'}</span>
            </div>
          </>
        )}
        {type === 'impulse' && (
          <div class="detail-grid">
            <span class="label">Path:</span>
            <span>{data.path}</span>
            <span class="label">Payload Type:</span>
            <span>{data.payload_type || 'None'}</span>
          </div>
        )}
      </div>
    </div>
  );
}
