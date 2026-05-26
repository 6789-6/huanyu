import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";
import { TRACKING } from "./config.js";

export async function createTracker(state) {
  state.ui.status.textContent = "加载识别模型...";
  const vision = await FilesetResolver.forVisionTasks(TRACKING.wasmUrl);

  const tracker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: TRACKING.modelUrl,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: TRACKING.maxHands,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.55
  });

  state.ui.status.textContent = "模型就绪";
  return tracker;
}

export async function startCamera(state) {
  if (state.running || state.starting) return;
  state.starting = true;

  state.ui.startBtn.disabled = true;
  state.ui.startBtn.textContent = "正在启动...";
  state.ui.status.textContent = "请求摄像头权限...";

  try {
    if (!state.tracker) {
      state.tracker = await createTracker(state);
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: state.preset.videoWidth },
        height: { ideal: state.preset.videoHeight },
        frameRate: { ideal: 24, max: 30 },
        facingMode: "user"
      },
      audio: false
    });

    state.video.srcObject = stream;
    await state.video.play();

    state.running = true;
    state.ui.status.textContent = "双手识别运行中";
    state.ui.startBtn.textContent = "识别运行中";
    state.ui.startBtn.disabled = true;
  } catch (err) {
    console.error(err);
    state.ui.status.textContent = "启动失败";
    state.ui.gesture.textContent = "请检查摄像头权限 / 浏览器 / localhost";
    state.ui.startBtn.textContent = "重新启动双手识别";
    state.ui.startBtn.disabled = false;
  } finally {
    state.starting = false;
  }
}
