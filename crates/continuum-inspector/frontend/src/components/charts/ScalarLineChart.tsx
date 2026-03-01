import { useEffect, useRef } from 'preact/hooks';
import * as d3 from 'd3';
import type { SignalHistoryEntry } from '../../types/ipc';
import { useContainerSize } from '../../hooks/useContainerSize';

interface ScalarLineChartProps {
  entries: SignalHistoryEntry[];
  label?: string;
}

/** D3 line chart for scalar signal history over time. Fills container. */
export function ScalarLineChart({ entries, label }: ScalarLineChartProps) {
  const { ref: containerRef, width, height } = useContainerSize();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || entries.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 10, right: 15, bottom: 30, left: 55 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;
    if (w <= 0 || h <= 0) return;

    const scalarEntries = entries.filter((e) => e.scalar != null);
    if (scalarEntries.length === 0) return;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const xExtent = d3.extent(scalarEntries, (d) => d.sim_time) as [number, number];
    const yExtent = d3.extent(scalarEntries, (d) => d.scalar!) as [number, number];
    const yPad = (yExtent[1] - yExtent[0]) * 0.05 || 1;

    const x = d3.scaleLinear().domain(xExtent).range([0, w]);
    const y = d3
      .scaleLinear()
      .domain([yExtent[0] - yPad, yExtent[1] + yPad])
      .range([h, 0]);

    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(Math.max(2, Math.floor(w / 80))))
      .selectAll('text')
      .attr('fill', '#999');

    g.append('g')
      .call(d3.axisLeft(y).ticks(Math.max(2, Math.floor(h / 40))))
      .selectAll('text')
      .attr('fill', '#999');

    g.selectAll('.domain, .tick line').attr('stroke', '#444');

    const line = d3
      .line<SignalHistoryEntry>()
      .x((d) => x(d.sim_time))
      .y((d) => y(d.scalar!))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(scalarEntries)
      .attr('fill', 'none')
      .attr('stroke', '#4fc3f7')
      .attr('stroke-width', 1.5)
      .attr('d', line);

    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 28)
      .attr('text-anchor', 'middle')
      .attr('fill', '#888')
      .attr('font-size', '10px')
      .text('sim time (s)');

    if (label) {
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -40)
        .attr('text-anchor', 'middle')
        .attr('fill', '#888')
        .attr('font-size', '10px')
        .text(label);
    }
  }, [entries, width, height, label]);

  if (entries.length === 0) {
    return <div class="chart-empty">No history data yet</div>;
  }

  return (
    <div ref={containerRef} style="flex:1;min-height:0;min-width:0;">
      <svg ref={svgRef} width={width} height={height} style="background:transparent;display:block;" />
    </div>
  );
}
