import { averagePoint, clamp, distance, hueGradient, lerp } from "./math.js";

export class Particle {
  constructor(index, state) {
    this.index = index;
    this.seed = Math.random() * 10000;
    this.layer = Math.random();
    this.heartT = Math.random() * Math.PI * 2;
    this.reset(state, true);
  }

  reset(state, randomPlace = false) {
    this.x = randomPlace ? Math.random() * state.width : state.width / 2;
    this.y = randomPlace ? Math.random() * state.height : state.height / 2;
    this.vx = (Math.random() - 0.5) * 2 * state.dpr;
    this.vy = (Math.random() - 0.5) * 2 * state.dpr;
    this.size = (Math.random() * 0.45 + 0.42) * state.dpr;
    this.energy = Math.random();
    this.offset = (Math.random() - 0.5) * 16;
  }

  update(t, state) {
    if (state.scene.kind === "heart" && state.scene.center) {
      this.applyHeart(t, state);
    } else if (state.scene.combo !== "none" && state.scene.controls.length >= 2) {
      this.applyIdle(t, state);
      this.applyCombo(t, state);
    } else {
      this.applyIdle(t, state);
      for (const c of state.scene.controls) {
        this.applyControl(c, t, state);
      }
    }

    this.vx *= state.scene.kind === "heart" ? 0.934 : 0.96;
    this.vy *= state.scene.kind === "heart" ? 0.934 : 0.96;

    const maxSpeed = state.scene.kind === "heart" ? 6.4 * state.dpr : 4.8 * state.dpr;
    const speed = Math.hypot(this.vx, this.vy);
    if (speed > maxSpeed) {
      this.vx = (this.vx / speed) * maxSpeed;
      this.vy = (this.vy / speed) * maxSpeed;
    }

    this.x += this.vx;
    this.y += this.vy;

    if (this.x < -40 * state.dpr || this.x > state.width + 40 * state.dpr || this.y < -40 * state.dpr || this.y > state.height + 40 * state.dpr) {
      this.x = clamp(this.x, 0, state.width);
      this.y = clamp(this.y, 0, state.height);
      this.vx *= -0.56;
      this.vy *= -0.56;
    }
  }

  applyIdle(t, state) {
    const cx = state.width * 0.5;
    const cy = state.height * 0.52;
    const dx = this.x - cx;
    const dy = this.y - cy;
    const d = Math.hypot(dx, dy) + 0.001;
    const swirl = 0.22 + Math.sin(t * 0.00038 + this.seed) * 0.10;

    this.vx += (-dy / d) * swirl * 0.042 * state.dpr;
    this.vy += (dx / d) * swirl * 0.042 * state.dpr;
    this.vx += (cx - this.x) * 0.000036;
    this.vy += (cy - this.y) * 0.000036;
  }

  applyHeart(t, state) {
    const center = state.scene.center;
    const scale = state.scene.scale;
    const beat = 1 + Math.sin(t * 0.0065) * 0.11 + Math.max(0, Math.sin(t * 0.013)) * 0.05;
    const localT = this.heartT + Math.sin(t * 0.0012 + this.seed) * 0.035;

    const hx = 16 * Math.pow(Math.sin(localT), 3);
    const hy = 13 * Math.cos(localT) - 5 * Math.cos(2 * localT) - 2 * Math.cos(3 * localT) - Math.cos(4 * localT);
    const nebula = this.layer < 0.72 ? 1 : 1.24 + this.layer * 0.52;

    const tx = center.x + hx * scale * 0.055 * beat * nebula + Math.sin(t * 0.004 + this.seed) * this.offset * state.dpr;
    const ty = center.y - hy * scale * 0.055 * beat * nebula + Math.cos(t * 0.004 + this.seed) * this.offset * state.dpr;

    const dx = tx - this.x;
    const dy = ty - this.y;
    const d = Math.hypot(dx, dy) + 0.001;

    const pull = this.layer < 0.72 ? 0.011 : 0.006;
    const spin = this.layer < 0.72 ? 0.030 : 0.062;
    this.vx += dx * pull + (-dy / d) * spin * state.dpr;
    this.vy += dy * pull + (dx / d) * spin * state.dpr;

    const pulse = Math.max(0, Math.sin(t * 0.013 + this.layer));
    this.vx += ((this.x - center.x) / d) * pulse * 0.10 * state.dpr;
    this.vy += ((this.y - center.y) / d) * pulse * 0.10 * state.dpr;
  }

  applyCombo(t, state) {
    const [ca, cb] = state.scene.controls;
    const a = ca.point;
    const b = cb.point;
    const mid = averagePoint([a, b]);
    const dxM = mid.x - this.x;
    const dyM = mid.y - this.y;
    const dM = Math.hypot(dxM, dyM) + 0.001;
    const intensity = state.effectIntensity;

    if (state.scene.combo === "dualPortal") {
      const bridge = distance(a, b) + 0.001;
      const phase = Math.atan2(b.y - a.y, b.x - a.x);
      const targetT = (this.index / Math.max(1, state.particles.length)) * Math.PI * 2 + t * 0.0026;
      const radius = bridge * 0.3 + Math.sin(this.seed + t * 0.004) * 22 * state.dpr;
      const tx = mid.x + Math.cos(targetT) * radius;
      const ty = mid.y + Math.sin(targetT) * radius * 0.42;
      const rx = Math.cos(phase) * (tx - mid.x) - Math.sin(phase) * (ty - mid.y) + mid.x;
      const ry = Math.sin(phase) * (tx - mid.x) + Math.cos(phase) * (ty - mid.y) + mid.y;

      this.vx += (rx - this.x) * 0.0054 * intensity;
      this.vy += (ry - this.y) * 0.0054 * intensity;
    }

    if (state.scene.combo === "gravityCollapse") {
      const gravity = Math.min(5.8 * state.dpr, (12800 * state.dpr) / (dM * dM));
      this.vx += (dxM / dM) * gravity * intensity;
      this.vy += (dyM / dM) * gravity * intensity;
      this.vx += (-dyM / dM) * 0.48 * state.dpr;
      this.vy += (dxM / dM) * 0.48 * state.dpr;
    }

    if (state.scene.combo === "laserBridge") {
      const abx = b.x - a.x;
      const aby = b.y - a.y;
      const len2 = abx * abx + aby * aby + 0.001;
      const u = clamp(((this.x - a.x) * abx + (this.y - a.y) * aby) / len2, 0, 1);
      const px = a.x + abx * u;
      const py = a.y + aby * u;
      const dx = px - this.x;
      const dy = py - this.y;
      const d = Math.hypot(dx, dy) + 0.001;
      const wave = Math.sin(u * Math.PI * 12 - t * 0.012 + this.seed) * 22 * state.dpr;
      const invLen = 1 / Math.sqrt(len2);
      const nx = -aby * invLen;
      const ny = abx * invLen;
      const pull = Math.max(0, 1 - d / (160 * state.dpr)) * 1.0 * state.dpr * intensity;

      this.vx += (dx / d) * pull + nx * wave * 0.00075;
      this.vy += (dy / d) * pull + ny * wave * 0.00075;
    }

    if (state.scene.combo === "wormhole") {
      const choice = this.index % 2 === 0 ? a : b;
      const dx = choice.x - this.x;
      const dy = choice.y - this.y;
      const d = Math.hypot(dx, dy) + 0.001;
      const spin = Math.max(0, 1 - d / (400 * state.dpr)) * 1.25 * state.dpr;
      this.vx += ((dx / d) * 0.46 * state.dpr + (-dy / d) * spin) * intensity;
      this.vy += ((dy / d) * 0.46 * state.dpr + (dx / d) * spin) * intensity;
    }
  }

  applyControl(control, t, state) {
    const p = control.point;
    const dx = p.x - this.x;
    const dy = p.y - this.y;
    const d = Math.hypot(dx, dy) + 0.001;
    const nx = dx / d;
    const ny = dy / d;
    const intensity = state.effectIntensity;

    if (control.type === "open") {
      const power = Math.max(0, 1 - d / (400 * state.dpr)) * 2.0 * state.dpr * intensity;
      this.vx -= nx * power;
      this.vy -= ny * power;
      this.vx += (-ny) * power * 0.2;
      this.vy += nx * power * 0.2;
    }

    if (control.type === "fist") {
      const ring = 64 * state.dpr + Math.sin(t * 0.004 + this.seed) * 26 * state.dpr;
      const angle = this.index * 0.047 + t * 0.0012 + control.handIndex;
      const tx = p.x + Math.cos(angle) * ring;
      const ty = p.y + Math.sin(angle) * ring;
      this.vx += (tx - this.x) * 0.0048 * intensity;
      this.vy += (ty - this.y) * 0.0048 * intensity;
    }

    if (control.type === "pinch") {
      const gravity = Math.min(4.0 * state.dpr, (7600 * state.dpr) / (d * d));
      const spin = Math.max(0, 1 - d / (470 * state.dpr)) * 1.1 * state.dpr;
      this.vx += (nx * gravity + (-ny) * spin) * intensity;
      this.vy += (ny * gravity + nx * spin) * intensity;
    }

    if (control.type === "peace") {
      const radius = 96 * state.dpr + Math.sin(this.index * 0.12 + t * 0.004) * 40 * state.dpr;
      const angle = this.index * 0.09 + t * 0.003 + control.handIndex * Math.PI;
      const strand = this.index % 2 === 0 ? 0 : Math.PI;
      const tx = p.x + Math.cos(angle + strand) * radius;
      const ty = p.y + Math.sin(angle + strand) * radius * 0.54;
      this.vx += (tx - this.x) * 0.0039 * intensity;
      this.vy += (ty - this.y) * 0.0039 * intensity;
    }

    if (control.type === "point") {
      const force = Math.max(0, 1 - d / (500 * state.dpr)) * 0.62 * state.dpr * intensity;
      this.vx += nx * force + Math.sin(d * 0.024 + t * 0.006) * 0.044 * state.dpr;
      this.vy += ny * force + Math.cos(d * 0.024 + t * 0.006) * 0.044 * state.dpr;
    }

    if (control.type === "portal") {
      const radius = 90 * state.dpr + Math.sin(this.seed + t * 0.006) * 26 * state.dpr;
      const angle = this.index * 0.064 + t * 0.0042;
      const tx = p.x + Math.cos(angle) * radius;
      const ty = p.y + Math.sin(angle) * radius;
      this.vx += (tx - this.x) * 0.005 * intensity;
      this.vy += (ty - this.y) * 0.005 * intensity;
    }

    if (control.type === "lightning") {
      const bolt = Math.sin(this.index * 1.7 + t * 0.03) > 0.18 ? 1 : -1;
      const force = Math.max(0, 1 - d / (480 * state.dpr)) * 0.88 * state.dpr * intensity;
      this.vx += nx * force + (-ny) * bolt * force * 1.18;
      this.vy += ny * force + nx * bolt * force * 1.18;
    }

    if (control.type === "fountain") {
      const lift = Math.max(0, 1 - d / (470 * state.dpr)) * 1.02 * state.dpr * intensity;
      const side = Math.sin(this.seed + t * 0.008) * 0.46 * state.dpr;
      this.vx += nx * lift * 0.4 + side;
      this.vy -= lift * 1.55;
    }

    if (control.type === "triangle") {
      const angle = ((this.index % 3) * Math.PI * 2) / 3 - Math.PI / 2 + t * 0.0008;
      const radius = 116 * state.dpr + Math.sin(t * 0.004 + this.seed) * 18 * state.dpr;
      const tx = p.x + Math.cos(angle) * radius;
      const ty = p.y + Math.sin(angle) * radius;
      this.vx += (tx - this.x) * 0.0046 * intensity;
      this.vy += (ty - this.y) * 0.0046 * intensity;
    }
  }

  draw(t, state) {
    const hue = hueGradient(state, this.x, this.y, t, this.layer, this.seed);
    const spriteIndex = Math.floor((hue / 360) * state.sprites.length) % state.sprites.length;
    const sprite = state.sprites[spriteIndex];

    const size = state.scene.kind === "heart"
      ? (4.3 + this.layer * 2.8 + this.energy * 1.3) * state.dpr
      : (3.1 + this.layer * 1.8 + this.energy) * state.dpr;

    state.ctx.globalAlpha = state.scene.kind === "heart" ? 0.92 : 0.82;
    state.ctx.drawImage(sprite, this.x - size, this.y - size, size * 2, size * 2);
    state.ctx.globalAlpha = 1;

    if (this.index % 26 === 0) {
      state.ctx.fillStyle = "rgba(255,255,255,0.84)";
      state.ctx.fillRect(this.x - 0.55 * state.dpr, this.y - 0.55 * state.dpr, 1.1 * state.dpr, 1.1 * state.dpr);
    }
  }
}
