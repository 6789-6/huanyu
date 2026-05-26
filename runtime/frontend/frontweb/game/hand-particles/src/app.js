import { PRESETS } from "./config.js";
import { Particle } from "./particles.js";
import { buildParticleSprites } from "./sprites.js";
import { updateSceneFromLandmarks } from "./gestures.js";
import { drawBackground, drawConnections, drawOverlays, initStars } from "./renderer.js";
import { startCamera } from "./tracker.js";

const canvas = document.querySelector("#scene");
const ctx = canvas.getContext("2d", { alpha: false });
const video = document.querySelector("#webcam");

const ui = {
  status: document.querySelector("#status"),
  fps: document.querySelector("#fps"),
  hands: document.querySelector("#hands"),
  particles: document.querySelector("#particles"),
  gesture: document.querySelector("#gesture"),
  mode: document.querySelector("#mode"),
  quality: document.querySelector("#quality"),
  intensity: document.querySelector("#intensity"),
  showSkeleton: document.querySelector("#showSkeleton"),
  showCamera: document.querySelector("#showCamera"),
  autoTune: document.querySelector("#autoTune"),
  startBtn: document.querySelector("#startBtn"),
  hudPanel: document.querySelector("#hudPanel"),
  hudToggle: document.querySelector("#hudToggle")
};

const state = {
  canvas,
  ctx,
  video,
  ui,

  presetName: "balanced",
  preset: PRESETS.balanced,
  dpr: Math.min(window.devicePixelRatio || 1, PRESETS.balanced.dprCap),
  width: 0,
  height: 0,

  particles: [],
  stars: [],
  sprites: [],
  frame: 0,

  tracker: null,
  running: false,
  starting: false,
  lastDetectTime: 0,
  latestResult: null,

  scene: {
    kind: "idle",
    combo: "none",
    controls: [],
    center: null,
    scale: 160,
    label: "未检测到手：星云待机",
    mode: "星云待机"
  },
  prevControlPoints: [],
  gestureHistory: [],
  stableGestureKey: "idle",

  showSkeleton: true,
  effectIntensity: 1,
  autoTune: true,
  lastAutoTuneAt: 0,

  fpsFrames: 0,
  fpsLast: performance.now(),
  fpsValue: 0
};

function resize() {
  state.dpr = Math.min(window.devicePixelRatio || 1, state.preset.dprCap);
  state.width = Math.floor(window.innerWidth * state.dpr);
  state.height = Math.floor(window.innerHeight * state.dpr);

  canvas.width = state.width;
  canvas.height = state.height;
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function initParticles() {
  state.sprites = buildParticleSprites(state.preset.particleSpriteSize, 96);
  state.particles = Array.from(
    { length: state.preset.particleCount },
    (_, i) => new Particle(i, state)
  );
  ui.particles.textContent = state.particles.length;
}

function applyPreset(name) {
  state.presetName = name;
  state.preset = { ...PRESETS[name] };
  state.showSkeleton = ui.showSkeleton.checked && state.preset.skeleton;

  resize();
  initParticles();
  initStars(state);

  ui.particles.textContent = state.particles.length;
}

function setHudCollapsed(collapsed) {
  if (!ui.hudPanel || !ui.hudToggle) return;
  ui.hudPanel.classList.toggle("is-collapsed", collapsed);
  ui.hudToggle.textContent = collapsed ? "SHOW" : "HIDE";
  ui.hudToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
  try {
    localStorage.setItem("huanyu-game-hud-collapsed", collapsed ? "1" : "0");
  } catch (error) {
    // Storage can be unavailable in strict privacy modes; the toggle still works.
  }
}

function autoTune() {
  if (!state.autoTune) return;
  if (performance.now() - state.lastAutoTuneAt < 4500) return;

  if (state.fpsValue > 0 && state.fpsValue < 26) {
    state.lastAutoTuneAt = performance.now();

    if (state.presetName === "showcase") {
      ui.quality.value = "cinematic";
      applyPreset("cinematic");
      ui.status.textContent = "自动降载：切到电影感";
    } else if (state.presetName === "cinematic") {
      ui.quality.value = "balanced";
      applyPreset("balanced");
      ui.status.textContent = "自动降载：切到均衡";
    } else if (state.presetName === "balanced") {
      ui.quality.value = "smooth";
      applyPreset("smooth");
      ui.status.textContent = "自动降载：切到流畅";
    }
  }
}

function detect(now) {
  if (!state.running || !state.tracker || state.video.readyState < 2) return;
  if (now - state.lastDetectTime < state.preset.detectInterval) return;

  state.lastDetectTime = now;
  state.latestResult = state.tracker.detectForVideo(state.video, now);
  updateSceneFromLandmarks(state.latestResult, state);
}

function animate(t) {
  state.frame++;
  state.fpsFrames++;

  if (t - state.fpsLast >= 500) {
    state.fpsValue = Math.round((state.fpsFrames * 1000) / (t - state.fpsLast));
    state.fpsFrames = 0;
    state.fpsLast = t;
    ui.fps.textContent = state.fpsValue;
    autoTune();
  }

  detect(t);

  drawBackground(t, state);

  ctx.globalCompositeOperation = "lighter";
  for (const p of state.particles) p.update(t, state);
  drawConnections(state);
  for (const p of state.particles) p.draw(t, state);

  ctx.globalAlpha = 1;
  drawOverlays(t, state);

  requestAnimationFrame(animate);
}

ui.startBtn.addEventListener("click", () => startCamera(state));

if (ui.hudToggle) {
  let savedHudState = "0";
  try {
    savedHudState = localStorage.getItem("huanyu-game-hud-collapsed") || "0";
  } catch (error) {
    savedHudState = "0";
  }
  setHudCollapsed(savedHudState === "1");
  ui.hudToggle.addEventListener("click", () => {
    setHudCollapsed(!ui.hudPanel.classList.contains("is-collapsed"));
  });
}

ui.quality.addEventListener("change", () => {
  applyPreset(ui.quality.value);
});

ui.intensity.addEventListener("input", () => {
  state.effectIntensity = Number(ui.intensity.value);
});

ui.showSkeleton.addEventListener("change", () => {
  state.showSkeleton = ui.showSkeleton.checked && state.preset.skeleton;
});

ui.showCamera.addEventListener("change", () => {
  video.classList.toggle("hidden", !ui.showCamera.checked);
});

ui.autoTune.addEventListener("change", () => {
  state.autoTune = ui.autoTune.checked;
});

window.addEventListener("resize", () => {
  resize();
  initStars(state);
});

applyPreset("balanced");
requestAnimationFrame(animate);
