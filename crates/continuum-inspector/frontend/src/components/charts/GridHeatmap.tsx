import { useEffect, useRef } from 'preact/hooks';
import * as d3 from 'd3';
import type { FieldSampleEntry } from '../../types/ipc';

interface GridHeatmapProps {
  samples: FieldSampleEntry[];
  width?: number;
  height?: number;
  label?: string;
}

/** D3 heatmap for grid2d field samples. Uses XY positions as cell coordinates. */
export function GridHeatmap({
  samples,
  width = 400,
  height = 300,
  label,
}: GridHeatmapProps) {
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

    const scalarSamples = samples.filter((s) => s.scalar != null);
    if (scalarSamples.length === 0) return;

    // Determine unique X and Y values for grid cells
    const xVals = [...new Set(scalarSamples.map((s) => s.position[0]))].sort(
      (a, b) => a - b
    );
    const yVals = [...new Set(scalarSamples.map((s) => s.position[1]))].sort(
      (a, b) => a - b
    );

    const colorExtent = d3.extent(scalarSamples, (d) => d.scalar!) as [number, number];
    const color = d3
      .scaleSequential(d3.interpolateInferno)
      .domain(colorExtent);

    // Band scales for grid cells
    const x = d3
      .scaleBand<number>()
      .domain(xVals)
      .range([0, w])
      .padding(0.02);
    const y = d3
      .scaleBand<number>()
      .domain(yVals)
      .range([h, 0])
      .padding(0.02);

    // Draw cells
    g.selectAll('rect.cell')
      .data(scalarSamples)
      .join('rect')
      .attr('class', 'cell')
      .attr('x', (d) => x(d.position[0]) ?? 0)
      .attr('y', (d) => y(d.position[1]) ?? 0)
      .attr('width', x.bandwidth())
      .attr('height', y.bandwidth())
      .attr('fill', (d) => color(d.scalar!));

    // Axes (show subset of ticks for readability)
    const xTickCount = Math.min(xVals.length, 8);
    const yTickCount = Math.min(yVals.length, 8);
    const xTickValues = xVals.filter(
      (_, i) => i % Math.ceil(xVals.length / xTickCount) === 0
    );
    const yTickValues = yVals.filter(
      (_, i) => i % Math.ceil(yVals.length / yTickCount) === 0
    );

    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).tickValues(xTickValues))
      .selectAll('text')
      .attr('fill', '#999')
      .attr('font-size', '9px');

    g.append('g')
      .call(d3.axisLeft(y).tickValues(yTickValues))
      .selectAll('text')
      .attr('fill', '#999')
      .attr('font-size', '9px');

    g.selectAll('.domain, .tick line').attr('stroke', '#444');

    // Color legend
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
      .attr('id', 'heatmap-gradient')
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
        .attr('stop-color', d3.interpolateInferno(t));
    }

    const legend = g.append('g').attr('transform', `translate(${legendX},10)`);
    legend
      .append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#heatmap-gradient)');

    legend
      .append('g')
      .attr('transform', `translate(${legendWidth},0)`)
      .call(d3.axisRight(legendScale).ticks(4))
      .selectAll('text')
      .attr('fill', '#999')
      .attr('font-size', '9px');

    legend.selectAll('.domain, .tick line').attr('stroke', '#444');

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
