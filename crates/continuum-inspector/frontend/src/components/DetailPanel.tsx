interface DetailPanelProps {
  selectedItem: any;
  ws: any;
}

export function DetailPanel({ selectedItem }: DetailPanelProps) {
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
      {data.doc && (
        <div class="detail-doc">{data.doc}</div>
      )}
      <div class="detail-content">
        {type === 'signal' && (
          <div class="detail-grid">
            <span class="label">ID:</span><span>{data.id}</span>
            <span class="label">Type:</span><span>{data.value_type}</span>
            {data.unit && <><span class="label">Unit:</span><span>{data.unit}</span></>}
            {data.stratum && <><span class="label">Stratum:</span><span>{data.stratum}</span></>}
            {data.range && <><span class="label">Range:</span><span>{data.range[0]} … {data.range[1]}</span></>}
          </div>
        )}
        {type === 'field' && (
          <div class="detail-grid">
            <span class="label">ID:</span><span>{data.id}</span>
            <span class="label">Topology:</span><span>{data.topology}</span>
            <span class="label">Type:</span><span>{data.value_type}</span>
            {data.unit && <><span class="label">Unit:</span><span>{data.unit}</span></>}
            {data.range && <><span class="label">Range:</span><span>{data.range[0]} … {data.range[1]}</span></>}
          </div>
        )}
        {type === 'entity' && (
          <div class="detail-grid">
            <span class="label">ID:</span><span>{data.id}</span>
            {data.count_bounds && <><span class="label">Count Bounds:</span><span>{data.count_bounds[0]} … {data.count_bounds[1]}</span></>}
          </div>
        )}
        {type === 'chronicle' && (
          <>
            <div class="detail-grid">
              <span class="label">Event:</span><span>{data.name}</span>
              <span class="label">Tick:</span><span>{data.tick}</span>
              <span class="label">Era:</span><span>{data.era}</span>
              <span class="label">Sim Time:</span><span>{data.sim_time.toFixed(2)}s</span>
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
              <span class="label">Signal:</span><span>{data.signal_id}</span>
              <span class="label">Severity:</span><span class={`severity-${data.severity}`}>{data.severity.toUpperCase()}</span>
              <span class="label">Message:</span><span>{data.message}</span>
              <span class="label">Tick:</span><span>{data.tick}</span>
              <span class="label">Era:</span><span>{data.era}</span>
              <span class="label">Sim Time:</span><span>{data.sim_time.toFixed(2)}s</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
