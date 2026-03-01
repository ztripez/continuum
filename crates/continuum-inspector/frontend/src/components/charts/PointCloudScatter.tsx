import { useEffect, useRef } from 'preact/hooks';
import * as d3 from 'd3';
import type { FieldSampleEntry } from '../../types/ipc';

interface PointCloudScatterProps {
  samples: FieldSampleEntry[];
  width?: number;
  height?: number;
  label?: string;
}

/** D3 scatter plot for point cloud field samples (XY projection, color = value). */
export function PointCloudScatter({
  samples,
  width = 400,
  height = 300,
  label,
}: PointCloudScatterProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || samples.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 10, right: 60, bottom: 30, left: 50 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // XY projection
    const xExtent = d3.extent(samples, (d) => d.position[0]) as [number, number];
    const yExtent = d3.extent(samples, (d) => d.position[1]) as [number, number];

    const x = d3.scaleLinear().domain(xExtent).range([0, w]).nice();
    const y = d3.scaleLinear().domain(yExtent).range([h, 0]).nice();

    // Color scale from scalar values
    const scalarSamples = samples.filter((s) => s.scalar != null);
    const colorExtent = d3.extent(scalarSamples, (d) => d.scalar!) as [number, number];
    const color = d3
      .scaleSequential(d3.interpolateViridis)
      .domain(colorExtent[0] !== undefined ? colorExtent : [0, 1]);

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

    g.selectAll('.domain, .tick line').attr('stroke', '#444');

    // Points
    g.selectAll('circle')
      .data(samples)
      .join('circle')
      .attr('cx', (d) => x(d.position[0]))
      .attr('cy', (d) => y(d.position[1]))
      .attr('r', 3)
      .attr('fill', (d) => (d.scalar != null ? color(d.scalar) : '#888'))
      .attr('opacity', 0.8);

    // Color legend
    if (scalarSamples.length > 0) {
      const legendHeight = h - 20;
      const legendWidth = 12;
      const legendX = w + 10;

      const legendScale = d3
        .scaleLinear()
        .domain(colorExtent)
        .range([legendHeight, 0]);

      const defs = svg.append('defs');
      const gradient = defs
        .append('linearGradient')
        .attr('id', 'scatter-gradient')
        .attr('x1', '0%')
        .attr('y1', '100%')
        .attr('x2', '0%')
        .attr('y2', '0%');

      const nStops = 10;
      for (let i = 0; i <= nStops; i++) {
        const t = i / nStops;
        gradient
          .append('stop')
          .attr('offset', `${t * 100}%`)
          .attr('stop-color', d3.interpolateViridis(t));
      }

      const legend = g.append('g').attr('transform', `translate(${legendX},10)`);

      legend
        .append('rect')
        .attr('width', legendWidth)
        .attr('height', legendHeight)
        .style('fill', 'url(#scatter-gradient)');

      legend
        .append('g')
        .attr('transform', `translate(${legendWidth},0)`)
        .call(d3.axisRight(legendScale).ticks(4))
        .selectAll('text')
        .attr('fill', '#999')
        .attr('font-size', '9px');

      legend.selectAll('.domain, .tick line').attr('stroke', '#444');
    }

    // Label
    if (label) {
      g.append('text')
        .attr('x', w / 2)
        .attr('y', -2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#888')
        .attr('font-size', '10px')
        .text(label);
    }
  }, [samples, width, height, label]);

  if (samples.length === 0) {
    return <div class="chart-empty">No field samples</div>;
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
