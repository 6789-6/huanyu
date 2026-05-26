import { GESTURE_NAMES } from "./config.js";
import { averagePoint, distance, smoothPoint, toCanvasPoint } from "./math.js";

function fingerStates(lm, state) {
  const index = lm[8].y < lm[6].y - 0.025;
  const middle = lm[12].y < lm[10].y - 0.025;
  const ring = lm[16].y < lm[14].y - 0.02;
  const pinky = lm[20].y < lm[18].y - 0.02;

  const thumbTip = toCanvasPoint(lm[4], state);
  const thumbIp = toCanvasPoint(lm[3], state);
  const indexMcp = toCanvasPoint(lm[5], state);
  const thumb = distance(thumbTip, indexMcp) > distance(thumbIp, indexMcp) + 13 * state.dpr;

  return { thumb, index, middle, ring, pinky };
}

export function classifyHand(lm, handIndex, state) {
  const fingers = fingerStates(lm, state);
  const openCount = Object.values(fingers).filter(Boolean).length;

  const wrist = toCanvasPoint(lm[0], state);
  const thumbTip = toCanvasPoint(lm[4], state);
  const indexTip = toCanvasPoint(lm[8], state);
  const middleTip = toCanvasPoint(lm[12], state);
  const ringTip = toCanvasPoint(lm[16], state);
  const palm = averagePoint([lm[0], lm[5], lm[9], lm[13], lm[17]].map((p) => toCanvasPoint(p, state)));

  const pinch = distance(thumbTip, indexTip) < Math.min(state.width, state.height) * 0.052;
  const okPortal = pinch && fingers.middle && fingers.ring && fingers.pinky;
  const peace = fingers.index && fingers.middle && !fingers.ring && !fingers.pinky;
  const rock = fingers.index && !fingers.middle && !fingers.ring && fingers.pinky;
  const thumbsUp = fingers.thumb && !fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky;
  const triangle = fingers.index && fingers.middle && fingers.ring && !fingers.pinky;
  const fist = openCount <= 1;
  const openPalm = openCount >= 4;

  if (okPortal) return { type: "portal", label: "OK手势：星门旋涡", point: averagePoint([thumbTip, indexTip]), handIndex };
  if (pinch) return { type: "pinch", label: "捏合：黑洞吸积盘", point: averagePoint([thumbTip, indexTip]), handIndex };
  if (rock) return { type: "lightning", label: "摇滚手势：闪电风暴", point: palm, handIndex };
  if (thumbsUp) return { type: "fountain", label: "点赞：粒子喷泉", point: thumbTip, handIndex };
  if (triangle) return { type: "triangle", label: "三指：三角星阵", point: averagePoint([indexTip, middleTip, ringTip]), handIndex };
  if (peace) return { type: "peace", label: "剪刀手：双螺旋流", point: averagePoint([indexTip, middleTip]), handIndex };
  if (fist) return { type: "fist", label: "握拳：能量核心", point: palm, handIndex };
  if (openPalm) return { type: "open", label: "张开手掌：超新星爆散", point: palm, handIndex };
  if (fingers.index) return { type: "point", label: "食指：霓虹牵引", point: indexTip, handIndex };

  return { type: "point", label: "手部移动：磁场扰动", point: wrist, handIndex };
}

function detectTwoHandHeart(landmarks, state) {
  if (!landmarks || landmarks.length < 2) return null;

  const a = landmarks[0];
  const b = landmarks[1];

  const aIndex = toCanvasPoint(a[8], state);
  const bIndex = toCanvasPoint(b[8], state);
  const aThumb = toCanvasPoint(a[4], state);
  const bThumb = toCanvasPoint(b[4], state);
  const aWrist = toCanvasPoint(a[0], state);
  const bWrist = toCanvasPoint(b[0], state);

  const close = Math.min(state.width, state.height) * 0.12;
  const directIndexClose = distance(aIndex, bIndex) < close;
  const directThumbClose = distance(aThumb, bThumb) < close;
  const crossClose =
    distance(aIndex, bThumb) < close * 0.92 &&
    distance(bIndex, aThumb) < close * 0.92;

  const topPair = directIndexClose ? averagePoint([aIndex, bIndex]) : averagePoint([aIndex, bThumb]);
  const bottomPair = directThumbClose ? averagePoint([aThumb, bThumb]) : averagePoint([bIndex, aThumb]);
  const verticalGap = Math.abs(topPair.y - bottomPair.y);
  const handsApart = distance(aWrist, bWrist) > Math.min(state.width, state.height) * 0.12;

  if (!(((directIndexClose && directThumbClose) || crossClose) && verticalGap > 16 * state.dpr && handsApart)) {
    return null;
  }

  const center = averagePoint([aIndex, bIndex, aThumb, bThumb]);
  const fingerSpread = Math.max(
    distance(aIndex, aThumb),
    distance(bIndex, bThumb),
    distance(aIndex, bIndex),
    distance(aThumb, bThumb)
  );
  const wristSpread = distance(aWrist, bWrist);
  const scale = Math.max(125 * state.dpr, Math.min(285 * state.dpr, Math.max(fingerSpread * 1.75, wristSpread * 0.65, 135 * state.dpr)));

  return { center, scale };
}

function stableSceneKey(scene) {
  if (scene.kind === "heart") return "heart";
  if (scene.combo !== "none") return scene.combo;
  if (scene.controls.length === 0) return "idle";
  return scene.controls.map((c) => c.type).sort().join("+");
}

function pushHistory(state, scene) {
  const key = stableSceneKey(scene);
  state.gestureHistory.push(key);
  if (state.gestureHistory.length > 5) state.gestureHistory.shift();

  const counts = new Map();
  for (const item of state.gestureHistory) counts.set(item, (counts.get(item) || 0) + 1);

  let bestKey = key;
  let bestCount = 0;
  for (const [item, count] of counts) {
    if (count > bestCount) {
      bestKey = item;
      bestCount = count;
    }
  }

  if (bestCount >= 3 || state.gestureHistory.length < 3) {
    state.stableGestureKey = bestKey;
  }
}

export function updateSceneFromLandmarks(results, state) {
  const landmarks = results.landmarks || [];
  state.ui.hands.textContent = landmarks.length;

  const heart = detectTwoHandHeart(landmarks, state);
  let nextScene;

  if (heart) {
    nextScene = {
      kind: "heart",
      combo: "none",
      controls: [],
      center: smoothPoint(state.scene.center, heart.center, 0.42),
      scale: heart.scale,
      label: "双手爱心：跳动粒子爱心星云",
      mode: GESTURE_NAMES.heart
    };
  } else {
    const controls = landmarks.map((lm, i) => classifyHand(lm, i, state)).slice(0, 2);

    for (let i = 0; i < controls.length; i++) {
      controls[i].point = smoothPoint(state.prevControlPoints[i], controls[i].point, 0.38);
    }
    state.prevControlPoints = controls.map((c) => c.point);

    if (controls.length === 0) {
      nextScene = {
        kind: "idle",
        combo: "none",
        controls: [],
        center: null,
        scale: 160,
        label: "未检测到手：星云待机",
        mode: GESTURE_NAMES.idle
      };
    } else if (controls.length === 1) {
      nextScene = {
        kind: controls[0].type,
        combo: "none",
        controls,
        center: controls[0].point,
        scale: 160,
        label: controls[0].label,
        mode: GESTURE_NAMES[controls[0].type] || GESTURE_NAMES.idle
      };
    } else {
      const types = controls.map((c) => c.type).sort().join("+");
      let combo = "dualField";
      let label = `双手控制：${controls[0].label.split("：")[0]} + ${controls[1].label.split("：")[0]}`;
      let mode = GESTURE_NAMES.dualField;

      if (types === "open+open") {
        combo = "dualPortal";
        label = "双掌展开：星际传送门";
        mode = GESTURE_NAMES.dualPortal;
      } else if (types === "fist+fist") {
        combo = "gravityCollapse";
        label = "双拳合力：引力坍缩核心";
        mode = GESTURE_NAMES.gravityCollapse;
      } else if (types === "point+point") {
        combo = "laserBridge";
        label = "双指相对：激光粒子桥";
        mode = GESTURE_NAMES.laserBridge;
      } else if (types === "portal+portal") {
        combo = "wormhole";
        label = "双 OK：虫洞隧道";
        mode = GESTURE_NAMES.wormhole;
      }

      nextScene = {
        kind: "dual",
        combo,
        controls,
        center: averagePoint(controls.map((c) => c.point)),
        scale: 180,
        label,
        mode
      };
    }
  }

  pushHistory(state, nextScene);

  // 防抖：双手组合和爱心会稳定得更好。视觉控制点仍使用最新点，避免延迟感太大。
  state.scene = nextScene;
  state.ui.gesture.textContent = nextScene.label;
  state.ui.mode.textContent = nextScene.mode;
}
