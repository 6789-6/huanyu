import { HAND_CONNECTIONS } from "./config.js";
import { averagePoint, clamp, toCanvasPoint } from "./math.js";

export function initStars(state) {
  state.stars = Array.from({ length: state.preset.starCount }, () => ({
    x: Math.random() * state.width,
    y: Math.random() * state.height,
    z: Math.random() * 0.8 + 0.2,
    tw: Math.random() * Math.PI * 2
  }));
}

export function drawBackground(t, state) {
  const { ctx, width, height } = state;
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = state.scene.kind === "heart" ? "rgba(2, 0, 10, 0.18)" : "rgba(0, 1, 8, 0.205)";
  ctx.fillRect(0, 0, width, height);

  const cx = state.scene.kind === "heart" && state.scene.center
    ? state.scene.center.x
    : width * 0.48 + Math.sin(t * 0.00018) * width * 0.08;
  const cy = state.scene.kind === "heart" && state.scene.center
    ? state.scene.center.y
    : height * 0.52 + Math.cos(t * 0.00021) * height * 0.08;

  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(width, height) * 0.88);

  if (state.scene.kind === "heart") {
    grad.addColorStop(0, "rgba(251,113,133,.20)");
    grad.addColorStop(0.24, "rgba(240,171,252,.12)");
    grad.addColorStop(0.52, "rgba(167,139,250,.07)");
    grad.addColorStop(0.76, "rgba(103,232,249,.035)");
    grad.addColorStop(1, "rgba(0,0,0,0)");
  } else {
    grad.addColorStop(0, "rgba(56,189,248,.078)");
    grad.addColorStop(0.28, "rgba(59,130,246,.052)");
    grad.addColorStop(0.55, "rgba(124,58,237,.048)");
    grad.addColorStop(0.78, "rgba(236,72,153,.026)");
    grad.addColorStop(1, "rgba(0,0,0,0)");
  }

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  drawStars(t, state);
  drawSpaceFog(t, state);
}

function drawStars(t, state) {
  const { ctx } = state;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  for (const s of state.stars) {
    const alpha = 0.18 + Math.sin(t * 0.0015 + s.tw) * 0.12 + s.z * 0.27;
    ctx.fillStyle = state.scene.kind === "heart"
      ? `rgba(255,210,230,${alpha})`
      : `rgba(210,245,255,${alpha})`;

    const size = Math.max(1, s.z * 1.65 * state.dpr);
    ctx.fillRect(s.x, s.y, size, size);

    s.y += s.z * 0.055 * state.dpr;
    s.x += Math.sin(t * 0.0004 + s.tw) * 0.035 * state.dpr;

    if (s.y > state.height) {
      s.y = 0;
      s.x = Math.random() * state.width;
    }
  }

  ctx.restore();
}

function drawSpaceFog(t, state) {
  const { ctx, width, height } = state;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  const bandY = height * 0.58 + Math.sin(t * 0.00018) * height * 0.05;
  const band = ctx.createLinearGradient(0, bandY - height * 0.28, width, bandY + height * 0.18);
  band.addColorStop(0, "rgba(0,0,0,0)");
  band.addColorStop(0.25, "rgba(56,189,248,0.018)");
  band.addColorStop(0.5, "rgba(167,139,250,0.038)");
  band.addColorStop(0.75, "rgba(244,114,182,0.018)");
  band.addColorStop(1, "rgba(0,0,0,0)");

  ctx.fillStyle = band;
  ctx.translate(width * 0.5, height * 0.5);
  ctx.rotate(-0.18);
  ctx.fillRect(-width, -height * 0.18, width * 2, height * 0.34);
  ctx.restore();
}

export function drawConnections(state) {
  if (!state.preset.lineBudget || state.preset.lineBudget <= 0) return;
  if (state.frame % state.preset.connectionFrameSkip !== 0) return;

  const { ctx } = state;
  const threshold = state.preset.linkDistance * state.dpr;
  const threshold2 = threshold * threshold;
  let drawn = 0;

  ctx.beginPath();
  const stepI = state.particles.length > 2500 ? 7 : state.particles.length > 1500 ? 5 : 4;
  const stepJ = state.particles.length > 2500 ? 18 : state.particles.length > 1500 ? 14 : 11;

  for (let i = 0; i < state.particles.length; i += stepI) {
    const a = state.particles[i];

    for (let j = i + 10; j < state.particles.length && j < i + 120; j += stepJ) {
      if (drawn >= state.preset.lineBudget) break;
      const b = state.particles[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const d2 = dx * dx + dy * dy;

      if (d2 < threshold2) {
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        drawn++;
      }
    }

    if (drawn >= state.preset.lineBudget) break;
  }

  ctx.lineWidth = 0.48 * state.dpr;
  const line = ctx.createLinearGradient(0, 0, state.width, state.height);
  if (state.scene.kind === "heart") {
    line.addColorStop(0, "rgba(255,150,190,.11)");
    line.addColorStop(0.5, "rgba(255,255,255,.075)");
    line.addColorStop(1, "rgba(103,232,249,.10)");
  } else {
    line.addColorStop(0, "rgba(103,232,249,.095)");
    line.addColorStop(0.5, "rgba(167,139,250,.09)");
    line.addColorStop(1, "rgba(251,113,133,.085)");
  }

  ctx.strokeStyle = line;
  ctx.stroke();
}

export function drawOverlays(t, state) {
  drawComboEffects(t, state);
  drawRings(t, state);
  drawHeartGuide(t, state);
  drawHandSkeleton(state);
}

function drawComboEffects(t, state) {
  if (state.scene.combo === "none" || state.scene.controls.length < 2) return;

  const { ctx } = state;
  const a = state.scene.controls[0].point;
  const b = state.scene.controls[1].point;
  const mid = averagePoint([a, b]);

  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  if (state.scene.combo === "laserBridge") {
    const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
    grad.addColorStop(0, "rgba(103,232,249,.15)");
    grad.addColorStop(0.5, "rgba(255,255,255,.84)");
    grad.addColorStop(1, "rgba(240,171,252,.15)");

    for (let i = 0; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      const wobble = Math.sin(t * 0.011 + i) * 24 * state.dpr;
      ctx.quadraticCurveTo(mid.x, mid.y + wobble, b.x, b.y);
      ctx.lineWidth = (5 - i) * 0.55 * state.dpr;
      ctx.strokeStyle = grad;
      ctx.stroke();
    }
  }

  if (state.scene.combo === "dualPortal" || state.scene.combo === "wormhole") {
    const hue = state.scene.combo === "wormhole" ? 305 : 195;

    for (let i = 0; i < 5; i++) {
      const r = (52 + i * 34 + Math.sin(t * 0.005 + i) * 12) * state.dpr;
      ctx.beginPath();
      ctx.ellipse(
        mid.x,
        mid.y,
        r * 1.45,
        r * 0.56,
        Math.atan2(b.y - a.y, b.x - a.x) + t * 0.0015,
        0,
        Math.PI * 2
      );
      ctx.lineWidth = 1.3 * state.dpr;
      ctx.strokeStyle = `hsla(${hue + i * 18},100%,70%,${0.42 - i * 0.055})`;
      ctx.stroke();
    }
  }

  if (state.scene.combo === "gravityCollapse") {
    const pulse = 1 + Math.sin(t * 0.014) * 0.12;
    const glow = ctx.createRadialGradient(mid.x, mid.y, 0, mid.x, mid.y, 220 * state.dpr * pulse);
    glow.addColorStop(0, "rgba(255,255,255,.68)");
    glow.addColorStop(0.18, "rgba(167,139,250,.22)");
    glow.addColorStop(0.6, "rgba(56,189,248,.052)");
    glow.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(mid.x, mid.y, 220 * state.dpr * pulse, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function drawRings(t, state) {
  if (state.scene.kind === "heart" || !state.scene.controls.length) return;

  const { ctx } = state;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  for (const control of state.scene.controls) {
    const p = control.point;
    const baseHue =
      control.type === "pinch" ? 315 :
      control.type === "fist" ? 270 :
      control.type === "open" ? 185 :
      control.type === "peace" ? 210 :
      control.type === "lightning" ? 48 :
      control.type === "fountain" ? 165 :
      control.type === "triangle" ? 285 :
      control.type === "portal" ? 205 : 145;

    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(t * 0.0018 + control.handIndex * Math.PI * 0.4);

    for (let i = 0; i < 3; i++) {
      const pulse = Math.sin(t * 0.004 + i * 2.1 + control.handIndex) * 0.5 + 0.5;
      const rx = (44 + i * 28 + pulse * 17) * state.dpr;
      const ry = rx * (control.type === "pinch" ? 0.42 : 0.72);
      ctx.beginPath();
      ctx.ellipse(0, 0, rx, ry, i * 0.58, 0, Math.PI * 2);
      ctx.lineWidth = (1.35 + i * 0.3) * state.dpr;
      ctx.strokeStyle = `hsla(${baseHue + i * 22},100%,70%,${0.45 - i * 0.1})`;
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.arc(0, 0, 6.5 * state.dpr, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,.94)";
    ctx.fill();
    ctx.restore();
  }

  ctx.restore();
}

function drawHeartGuide(t, state) {
  if (state.scene.kind !== "heart" || !state.scene.center) return;

  const { ctx } = state;
  const beat = 1 + Math.sin(t * 0.0065) * 0.1 + Math.max(0, Math.sin(t * 0.013)) * 0.045;

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.translate(state.scene.center.x, state.scene.center.y);
  ctx.lineWidth = 1.3 * state.dpr;

  for (let layer = 0; layer < 3; layer++) {
    const scale = state.scene.scale * 0.055 * beat * (1 + layer * 0.18);
    ctx.beginPath();

    for (let i = 0; i <= 220; i++) {
      const a = (i / 220) * Math.PI * 2;
      const x = 16 * Math.pow(Math.sin(a), 3) * scale;
      const y = -(13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a)) * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.closePath();
    ctx.strokeStyle = layer === 0
      ? "rgba(255,255,255,.32)"
      : layer === 1
        ? "rgba(251,113,133,.26)"
        : "rgba(103,232,249,.20)";
    ctx.stroke();
  }

  ctx.restore();
}

function drawHandSkeleton(state) {
  if (!state.showSkeleton || !state.latestResult?.landmarks?.length) return;

  const { ctx } = state;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineWidth = 1.55 * state.dpr;

  for (let h = 0; h < state.latestResult.landmarks.length; h++) {
    const points = state.latestResult.landmarks[h].map((p) => toCanvasPoint(p, state));
    const hue = h === 0 ? 190 : 315;

    ctx.strokeStyle = `hsla(${hue},100%,70%,.55)`;
    ctx.beginPath();

    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.moveTo(points[a].x, points[a].y);
      ctx.lineTo(points[b].x, points[b].y);
    }

    ctx.stroke();

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      ctx.beginPath();
      ctx.arc(p.x, p.y, (i % 4 === 0 ? 3.9 : 2.4) * state.dpr, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${hue + i * 2},100%,${i % 4 === 0 ? 78 : 68}%,.9)`;
      ctx.fill();
    }
  }

  ctx.restore();
}
