export function buildParticleSprites(size = 24, count = 96) {
  const sprites = [];

  for (let i = 0; i < count; i++) {
    const hue = (i / count) * 360;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;

    const ctx = canvas.getContext("2d");
    const c = size / 2;
    const r = size / 2;

    const gradient = ctx.createRadialGradient(c - r * 0.18, c - r * 0.18, 0, c, c, r);
    gradient.addColorStop(0, "rgba(255,255,255,0.96)");
    gradient.addColorStop(0.18, `hsla(${hue}, 100%, 78%, 0.95)`);
    gradient.addColorStop(0.48, `hsla(${(hue + 36) % 360}, 100%, 62%, 0.70)`);
    gradient.addColorStop(0.78, `hsla(${(hue + 72) % 360}, 100%, 50%, 0.20)`);
    gradient.addColorStop(1, `hsla(${hue}, 100%, 48%, 0)`);

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    ctx.fillStyle = `hsla(${hue}, 100%, 76%, 0.96)`;
    ctx.beginPath();
    ctx.arc(c, c, Math.max(1.2, size * 0.1), 0, Math.PI * 2);
    ctx.fill();

    sprites.push(canvas);
  }

  return sprites;
}
