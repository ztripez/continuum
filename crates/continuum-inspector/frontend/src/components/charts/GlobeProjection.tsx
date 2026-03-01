import { useEffect, useRef } from 'preact/hooks';
import * as d3 from 'd3';
import type { FieldSampleEntry } from '../../types/ipc';

interface GlobeProjectionProps {
  samples: FieldSampleEntry[];
  width?: number;
  height?: number;
  label?: string;
}

/** D3 orthographic globe projection for sphere_surface field samples. 
 *  Positions are interpreted as [longitude, latitude, altitude] in degrees. */
export function GlobeProjection({
  samples,
  width = 400,
  height = 400,
  label,
}: GlobeProjectionProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || samples.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const size = Math.min(width, height);
    const radius = size / 2 - 20;

    const projection = d3
      .geoOrthographic()
      .scale(radius)
      .translate([width / 2, height / 2])
      .clipAngle(90);

    const path = d3.geoPath().projection(projection);

    // Globe background
    svg
      .append('circle')
      .attr('cx', width / 2)
      .attr('cy', height / 2)
      .attr('r', radius)
      .attr('fill', '#1a1a2e')
      .attr('stroke', '#444')
      .attr('stroke-width', 0.5);

    // Graticule
    const graticule = d3.geoGraticule().step([30, 30]);
    svg
      .append('path')
      .datum(graticule())
      .attr('d', path)
      .attr('fill', 'none')
      .attr('stroke', '#333')
      .attr('stroke-width', 0.3);

    // Color scale from scalar values
    const scalarSamples = samples.filter((s) => s.scalar != null);
    const colorExtent = d3.extent(scalarSamples, (d) => d.scalar!) as [number, number];
    const color = d3
      .scaleSequential(d3.interpolateViridis)
      .domain(colorExtent[0] !== undefined ? colorExtent : [0, 1]);

    // Plot sample points
    // Position: [x, y, z] where x=lon, y=lat for spherical fields
    const g = svg.append('g');

    for (const sample of samples) {
      const lon = sample.position[0];
      const lat = sample.position[1];
      const projected = projection([lon, lat]);
      if (!projected) continue; // Behind the globe

      const pointColor =
        sample.scalar != null ? color(sample.scalar) : '#888';

      g.append('circle')
        .attr('cx', projected[0])
        .attr('cy', projected[1])
        .attr('r', 2.5)
        .attr('fill', pointColor)
        .attr('opacity', 0.8)
        .attr('stroke', 'none');
    }

    // Label
    if (label) {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', 14)
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
