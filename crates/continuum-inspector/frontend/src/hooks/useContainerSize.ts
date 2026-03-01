import { useEffect, useRef, useState } from 'preact/hooks';

/** Tracks the content size of a container element via ResizeObserver. */
export function useContainerSize(fallbackWidth = 400, fallbackHeight = 300) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: fallbackWidth, height: fallbackHeight });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) {
        setSize({ width: Math.floor(width), height: Math.floor(height) });
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return { ref, ...size };
}
