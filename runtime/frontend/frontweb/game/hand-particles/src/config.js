export const TRACKING = {
  wasmUrl: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
  modelUrl:
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  maxHands: 2
};

export const PRESETS = {
  smooth: {
    label: "流畅",
    particleCount: 1100,
    starCount: 160,
    detectInterval: 130,
    videoWidth: 424,
    videoHeight: 240,
    dprCap: 0.85,
    lineBudget: 0,
    linkDistance: 0,
    particleSpriteSize: 22,
    connectionFrameSkip: 999,
    skeleton: false
  },
  balanced: {
    label: "均衡",
    particleCount: 1900,
    starCount: 240,
    detectInterval: 105,
    videoWidth: 480,
    videoHeight: 270,
    dprCap: 1.0,
    lineBudget: 420,
    linkDistance: 35,
    particleSpriteSize: 24,
    connectionFrameSkip: 2,
    skeleton: true
  },
  cinematic: {
    label: "电影感",
    particleCount: 2800,
    starCount: 320,
    detectInterval: 92,
    videoWidth: 640,
    videoHeight: 360,
    dprCap: 1.08,
    lineBudget: 720,
    linkDistance: 40,
    particleSpriteSize: 26,
    connectionFrameSkip: 2,
    skeleton: true
  },
  showcase: {
    label: "展示级",
    particleCount: 3800,
    starCount: 460,
    detectInterval: 82,
    videoWidth: 640,
    videoHeight: 360,
    dprCap: 1.15,
    lineBudget: 980,
    linkDistance: 44,
    particleSpriteSize: 28,
    connectionFrameSkip: 2,
    skeleton: true
  }
};

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

export const GESTURE_NAMES = {
  idle: "星云待机",
  open: "超新星爆散",
  fist: "能量核心",
  pinch: "黑洞吸积盘",
  peace: "双螺旋流",
  point: "霓虹牵引",
  portal: "星门旋涡",
  lightning: "闪电风暴",
  fountain: "粒子喷泉",
  triangle: "三角星阵",
  heart: "跳动爱心星云",
  dualPortal: "星际传送门",
  gravityCollapse: "引力坍缩核心",
  laserBridge: "激光粒子桥",
  wormhole: "虫洞隧道",
  dualField: "双手粒子磁场"
};
