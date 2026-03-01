import { useEffect, useRef } from 'preact/hooks';
import * as d3 from 'd3';
import type { SignalHistoryEntry } from '../../types/ipc';

interface ScalarLineChartProps {
  entries: SignalHistoryEntry[];
  width?: number;
  height?: number;
  label?: string;
}

/** D3 line chart for scalar signal history over time. */
export function ScalarLineChart({
  entries,
  width = 400,
  height = 200,
  label,
}: ScalarLineChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || entries.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 10, right: 15, bottom: 30, left: 50 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const scalarEntries = entries.filter((e) => e.scalar != null);
    if (scalarEntries.length === 0) return;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const xExtent = d3.extent(scalarEntries, (d) => d.sim_time) as [number, number];
    const yExtent = d3.extent(scalarEntries, (d) => d.scalar!) as [number, number];

    // Add 5% padding to y-axis
    const yPad = (yExtent[1] - yExtent[0]) * 0.05 || 1;

    const x = d3.scaleLinear().domain(xExtent).range([0, w]);
    const y = d3
      .scaleLinear()
      .domain([yExtent[0] - yPad, yExtent[1] + yPad])
      .range([h, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(5))
      .selectAll('text')
      .attr('fill', '#999');

    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text')
      .attr('fill', '#999');

    // Style axis lines
    g.selectAll('.domain, .tick line').attr('stroke', '#444');

    // Line
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

    // X-axis label
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 28)
      .attr('text-anchor', 'middle')
      .attr('fill', '#888')
      .attr('font-size', '10px')
      .text('sim time (s)');

    // Y-axis label
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
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style="background: transparent;"
    />
  );
}
