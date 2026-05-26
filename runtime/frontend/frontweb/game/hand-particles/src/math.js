export const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

export const lerp = (a, b, t) => a + (b - a) * t;

export function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function averagePoint(points) {
  const sum = points.reduce(
    (acc, p) => {
      acc.x += p.x;
      acc.y += p.y;
      return acc;
    },
    { x: 0, y: 0 }
  );

  return {
    x: sum.x / points.length,
    y: sum.y / points.length
  };
}

export function smoothPoint(prev, next, alpha = 0.34) {
  if (!prev) return next;
  return {
    x: lerp(prev.x, next.x, alpha),
    y: lerp(prev.y, next.y, alpha)
  };
}

export function toCanvasPoint(landmark, state) {
  return {
    x: (1 - landmark.x) * state.width,
    y: landmark.y * state.height,
    z: landmark.z || 0
  };
}

export function hueGradient(state, x, y, t, layer = 0, seed = 0) {
  const nx = x / Math.max(1, state.width);
  const ny = y / Math.max(1, state.height);

  let h = 195 + nx * 110 + Math.sin(t * 0.0012 + ny * 8 + seed) * 34;

  if (state.scene.kind === "heart") {
    h = 334 + Math.sin(t * 0.003 + layer * 5 + seed) * 22;
  } else if (state.scene.combo !== "none") {
    h += 48 * Math.sin(t * 0.002 + layer * 7);
  }

  return (h + 360) % 360;
}
