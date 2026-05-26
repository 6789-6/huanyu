(function () {
  "use strict";

  function getApiBase() {
    var configured = window.HUANYU_API_BASE || "";
    configured = String(configured).trim();
    if (configured) {
      return configured.replace(/\/+$/, "");
    }
    if (window.location && /^https?:$/.test(window.location.protocol) && window.location.port === "5000") {
      return window.location.origin;
    }
    if (window.location) {
      var host = window.location.hostname;
      if (host === "127.0.0.1" || host === "localhost") {
        return "http://127.0.0.1:5000";
      }
    }
    return window.location.origin;
  }

  var API_BASE = getApiBase();
  var state = {
    levels: [],
    selectedLevel: null,
    stream: null,
    recorder: null,
    chunks: [],
    recordingBlob: null,
    uploadedFile: null,
    recordingStartedAt: 0,
    recordingTimer: null,
    keypointTimer: null,
    keypointBusy: false,
    keypointPreview: null,
    keypointAnimationFrame: null,
    keypointAnimationStartedAt: 0,
    interactionLocked: false,
    progress: loadProgress()
  };

  function $(id) {
    return document.getElementById(id);
  }

  function loadProgress() {
    try {
      return JSON.parse(localStorage.getItem("huanyu-e2e-challenge-progress") || "{}");
    } catch (err) {
      return {};
    }
  }

  function saveProgress() {
    localStorage.setItem("huanyu-e2e-challenge-progress", JSON.stringify(state.progress));
  }

  function requestJson(path, options) {
    options = options || {};
    options.headers = window.HUANYU_AUTH ? window.HUANYU_AUTH.authHeaders(options.headers || {}) : (options.headers || {});
    return fetch(API_BASE + path, options || {}).then(function (response) {
      return response.json().then(function (payload) {
        if (!response.ok || payload.code >= 400) {
          throw new Error(payload.message || "请求失败");
        }
        return payload.data || {};
      });
    });
  }

  function scoreLabel(score) {
    if (score >= 80) return "已通过";
    if (score >= 50) return "待加强";
    return "需重练";
  }

  function renderStatus(data) {
    var box = $("model-status");
    if (!box) return;
    var direct = data.directE2E && data.directE2E.available;
    var s2g = data.twoStage && data.twoStage.s2gAvailable;
    var g2t = data.twoStage && data.twoStage.g2tAvailable;
    box.innerHTML = [
      statusRow("直接 E2E", direct ? "可用" : "缺失", direct ? "" : "fail"),
      statusRow("S2G 模型", s2g ? "可用" : "缺失", s2g ? "" : "warn"),
      statusRow("G2T 模型", g2t ? "可用" : "缺失", g2t ? "" : "warn"),
      statusRow("最佳 WER", data.bestWer || "69.42%", "warn")
    ].join("");
  }

  function statusRow(label, value, tone) {
    return '<div class="hy-status-item"><span>' + label + '</span><span class="hy-pill ' + (tone || "") + '">' + value + "</span></div>";
  }

  function renderIndexLevels() {
    var grid = $("level-grid");
    if (!grid) return;
    grid.innerHTML = state.levels.map(function (level) {
      var saved = state.progress[level.id] || {};
      var score = saved.score || 0;
      var href = "show.html?level=" + encodeURIComponent(level.id);
      return [
        '<a class="hy-card hy-level-card" href="' + href + '">',
        '<div class="hy-level-meta"><span class="hy-level-no">关卡 ' + level.id + '</span><span class="hy-pill ' + (score >= level.passScore ? "" : "warn") + '">' + (score ? score + " 分" : "未完成") + "</span></div>",
        "<h2>" + escapeHtml(level.title) + "</h2>",
        "<p>" + escapeHtml(level.targetChinese) + "</p>",
        '<div class="hy-progress"><span style="width:' + Math.min(score, 100) + '%"></span></div>',
        "</a>"
      ].join("");
    }).join("");
  }

  function renderLevelButtons() {
    var list = $("level-list");
    if (!list) return;
    list.innerHTML = state.levels.map(function (level) {
      var active = state.selectedLevel && state.selectedLevel.id === level.id;
      return '<button class="hy-level-button ' + (active ? "active" : "") + '" data-level-id="' + level.id + '">关卡 ' + level.id + " · " + escapeHtml(level.title) + "</button>";
    }).join("");

    Array.prototype.forEach.call(list.querySelectorAll("button"), function (button) {
      button.addEventListener("click", function () {
        selectLevel(button.getAttribute("data-level-id"));
      });
    });
  }

  function selectLevel(levelId) {
    var fallback = state.levels[0];
    state.selectedLevel = state.levels.find(function (item) { return String(item.id) === String(levelId); }) || fallback;
    if (!state.selectedLevel) return;
    var title = $("task-title");
    var target = $("target-chinese");
    var gloss = $("target-gloss");
    var pass = $("pass-score");
    if (title) title.textContent = "关卡 " + state.selectedLevel.id + " · " + state.selectedLevel.title;
    if (target) target.textContent = state.selectedLevel.targetChinese;
    if (gloss) gloss.textContent = state.selectedLevel.targetGloss;
    if (pass) pass.textContent = state.selectedLevel.passScore + " 分通过";
    renderLevelButtons();
    setResult({});
  }

  function escapeHtml(value) {
    return String(value || "").replace(/[&<>"']/g, function (char) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char];
    });
  }

  function setBusy(isBusy, text) {
    var submit = $("submit-btn");
    var status = $("run-status");
    if (submit) submit.disabled = isBusy;
    if (status) status.textContent = text || "";
  }

  function setCaptureStatus(status, timer, progress) {
    setText("capture-status", status || "准备录制");
    if (timer !== undefined) setText("capture-timer", timer);
    if (progress !== undefined) setText("capture-progress", progress);
  }

  function formatDuration(ms) {
    var totalSeconds = Math.max(0, Math.floor(ms / 1000));
    var minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
    var seconds = String(totalSeconds % 60).padStart(2, "0");
    return minutes + ":" + seconds;
  }

  function startRecordingClock() {
    stopRecordingClock();
    state.recordingStartedAt = Date.now();
    setCaptureStatus("正在录制", "00:00", "录制中");
    state.recordingTimer = window.setInterval(function () {
      setText("capture-timer", formatDuration(Date.now() - state.recordingStartedAt));
    }, 500);
  }

  function stopRecordingClock() {
    if (state.recordingTimer) {
      window.clearInterval(state.recordingTimer);
      state.recordingTimer = null;
    }
  }

  function setInteractionLocked(locked) {
    state.interactionLocked = Boolean(locked);
    ["start-camera-btn", "record-btn", "submit-btn", "video-file"].forEach(function (id) {
      var el = $(id);
      if (el) el.disabled = state.interactionLocked;
    });
    var file = document.querySelector(".hy-file");
    if (file) file.classList.toggle("is-disabled", state.interactionLocked);
  }

  function renderKeypoints(preview) {
    if (!preview || !preview.__animatedFrame) {
      state.keypointPreview = preview || null;
      if (preview && preview.hands && preview.hands.length) startKeypointAnimation();
      else stopKeypointAnimation();
    }
    var canvas = $("keypoint-canvas");
    var status = $("keypoint-status");
    var source = $("keypoint-source");
    if (!canvas || !canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    var width = canvas.width;
    var height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    drawKeypointBackground(ctx, width, height);
    if (!preview || !preview.hands || !preview.hands.length) {
      if (status) status.textContent = "等待识别";
      if (source) source.textContent = "未生成";
      drawKeypointEmpty(ctx, width, height);
      return;
    }
    var connections = preview.connections || [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
      [5, 9], [9, 13], [13, 17]
    ];
    var colors = ["#28f0d0", "#7aa7ff", "#f8d46a"];
    preview.hands.forEach(function (hand, index) {
      drawHandKeypoints(ctx, hand, connections, width, height, colors[index % colors.length]);
    });
    if (status) status.textContent = "识别完成";
    if (source) source.textContent = preview.source === "model-keypoints" ? "模型关键点" : "关键点预览";
  }

  function startKeypointAnimation() {
    if (state.keypointAnimationFrame) return;
    state.keypointAnimationStartedAt = performance.now();
    state.keypointAnimationFrame = window.requestAnimationFrame(animateKeypointPreview);
  }

  function stopKeypointAnimation() {
    if (state.keypointAnimationFrame) {
      window.cancelAnimationFrame(state.keypointAnimationFrame);
      state.keypointAnimationFrame = null;
    }
    state.keypointAnimationStartedAt = 0;
  }

  function animateKeypointPreview(now) {
    state.keypointAnimationFrame = null;
    if (!state.keypointPreview || !state.keypointPreview.hands || !state.keypointPreview.hands.length) return;
    renderKeypoints(applyKeypointMotion(state.keypointPreview, now));
    state.keypointAnimationFrame = window.requestAnimationFrame(animateKeypointPreview);
  }

  function applyKeypointMotion(preview, now) {
    var elapsed = Math.max(0, now - (state.keypointAnimationStartedAt || now));
    var phase = elapsed / 420;
    return {
      __animatedFrame: true,
      source: preview.source,
      connections: preview.connections,
      hands: (preview.hands || []).map(function (hand, handIndex) {
        return {
          label: hand.label,
          visible: hand.visible,
          points: (hand.points || []).map(function (point, pointIndex) {
            if (!point || !Number.isFinite(Number(point.x)) || !Number.isFinite(Number(point.y))) return point;
            var score = Number(point.score || 0);
            if (score <= 0) return point;
            var wave = phase + handIndex * 0.9 + pointIndex * 0.27;
            return {
              x: Math.max(0.02, Math.min(0.98, Number(point.x) + Math.sin(wave) * 0.0045)),
              y: Math.max(0.02, Math.min(0.98, Number(point.y) + Math.cos(wave * 0.86) * 0.0035)),
              score: score
            };
          })
        };
      })
    };
  }

  function drawKeypointBackground(ctx, width, height) {
    var gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, "#07141b");
    gradient.addColorStop(0.55, "#0b2330");
    gradient.addColorStop(1, "#071016");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(77, 246, 218, 0.12)";
    ctx.lineWidth = 1;
    for (var x = 24; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (var y = 24; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }

  function drawKeypointEmpty(ctx, width, height) {
    ctx.fillStyle = "rgba(227, 249, 255, 0.68)";
    ctx.font = "600 16px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("提交视频后显示识别关键点", width / 2, height / 2);
  }

  function drawHandKeypoints(ctx, hand, connections, width, height, color) {
    var points = hand.points || [];
    function valid(point) {
      return point && Number(point.score || 0) > 0 && Number.isFinite(Number(point.x)) && Number.isFinite(Number(point.y));
    }
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    connections.forEach(function (pair) {
      var a = points[pair[0]];
      var b = points[pair[1]];
      if (!valid(a) || !valid(b)) return;
      ctx.beginPath();
      ctx.moveTo(Number(a.x) * width, Number(a.y) * height);
      ctx.lineTo(Number(b.x) * width, Number(b.y) * height);
      ctx.stroke();
    });
    points.forEach(function (point, index) {
      if (!valid(point)) return;
      var radius = index === 0 ? 6 : 4.5;
      ctx.beginPath();
      ctx.fillStyle = index === 0 ? "#ffffff" : color;
      ctx.arc(Number(point.x) * width, Number(point.y) * height, radius, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  }

  function startKeypointTracking() {
    if (state.keypointTimer) return;
    setText("keypoint-status", "实时捕捉");
    setText("keypoint-source", "视频帧分析");
    captureKeypointFrame();
    state.keypointTimer = window.setInterval(captureKeypointFrame, 900);
  }

  function stopKeypointTracking() {
    if (state.keypointTimer) {
      window.clearInterval(state.keypointTimer);
      state.keypointTimer = null;
    }
    state.keypointBusy = false;
  }

  function captureKeypointFrame() {
    var video = $("camera-feed");
    if (!video || state.keypointBusy) return;
    if (!video.videoWidth || !video.videoHeight || video.readyState < 2) return;
    state.keypointBusy = true;
    var canvas = document.createElement("canvas");
    var maxWidth = 360;
    var ratio = video.videoHeight / video.videoWidth;
    canvas.width = maxWidth;
    canvas.height = Math.max(180, Math.round(maxWidth * ratio));
    var ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    fetch(API_BASE + "/api/recognition/realtime/frame", {
      method: "POST",
      headers: window.HUANYU_AUTH.authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ frameData: canvas.toDataURL("image/jpeg", 0.72) })
    })
      .then(function (response) { return response.json(); })
      .then(function (payload) {
        var data = payload.data || {};
        if (data.keypointPreview) {
          renderKeypoints(data.keypointPreview);
          setText("keypoint-status", data.fallback ? "关键点预览" : "实时关键点");
          setText("keypoint-source", data.fallback ? "备用估计" : "实时模型");
        }
      })
      .catch(function () {
        setText("keypoint-status", "关键点预览");
      })
      .finally(function () {
        state.keypointBusy = false;
      });
  }

  function setResult(data) {
    var score = data.score || 0;
    var resultPanel = document.querySelector(".hy-result");
    var scoreEl = $("score-value");
    if (scoreEl) scoreEl.textContent = score ? Math.round(score) + " 分" : "--";
    setText("recognized-text", data.result || "--");
    setText("recognized-gloss", data.gloss || "--");
    setText("model-name", data.model || "--");
    setText("pass-state", data.passed === undefined ? "--" : (data.passed ? "通过" : scoreLabel(score)));
    setText("feedback", data.feedback || "上传或录制一段手语视频后，系统会用 E2E 模型输出 gloss 并给出闯关评分。");
    if (resultPanel) {
      resultPanel.classList.remove("is-pass", "is-warn", "is-fail");
      if (data.passed === true) resultPanel.classList.add("is-pass");
      else if (data.passed === false && score >= 50) resultPanel.classList.add("is-warn");
      else if (data.passed === false) resultPanel.classList.add("is-fail");
    }
    renderKeypoints(data.keypointPreview);
  }

  function setText(id, value) {
    var el = $(id);
    if (el) el.textContent = value;
  }

  function initFileInput() {
    var input = $("video-file");
    var label = $("file-label");
    if (!input) return;
    input.addEventListener("change", function () {
      state.uploadedFile = input.files && input.files[0] ? input.files[0] : null;
      state.recordingBlob = null;
      if (label) label.textContent = state.uploadedFile ? state.uploadedFile.name : "选择视频";
      setCaptureStatus(state.uploadedFile ? "已选择视频" : "准备录制", "00:00", "待提交");
      var video = $("camera-feed");
      if (video && state.uploadedFile) {
        video.srcObject = null;
        video.src = URL.createObjectURL(state.uploadedFile);
        video.controls = true;
        video.play().catch(function () {});
        video.onloadeddata = startKeypointTracking;
        startKeypointTracking();
      }
    });
  }

  function initCameraControls() {
    var start = $("start-camera-btn");
    var record = $("record-btn");
    if (start) {
      start.addEventListener("click", function () {
        navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then(function (stream) {
          state.stream = stream;
          var video = $("camera-feed");
          video.srcObject = stream;
          video.controls = false;
          video.play();
          startKeypointTracking();
          if (record && window.MediaRecorder) record.disabled = false;
          setCaptureStatus("摄像头已打开", "00:00", "待录制");
        }).catch(function (err) {
          setBusy(false, "无法打开摄像头：" + err.message);
          setCaptureStatus("摄像头不可用", undefined, "待处理");
        });
      });
    }
    if (record) {
      record.addEventListener("click", toggleRecording);
      record.disabled = !window.MediaRecorder;
    }
  }

  function toggleRecording() {
    var record = $("record-btn");
    if (state.recorder && state.recorder.state === "recording") {
      state.recorder.stop();
      record.textContent = "开始录制";
      stopRecordingClock();
      return;
    }
    if (!state.stream) {
      setBusy(false, "请先打开摄像头。");
      setCaptureStatus("等待摄像头", undefined, "待录制");
      return;
    }
    state.chunks = [];
    state.recorder = new MediaRecorder(state.stream, { mimeType: pickMimeType() });
    state.recorder.ondataavailable = function (event) {
      if (event.data && event.data.size > 0) state.chunks.push(event.data);
    };
    state.recorder.onstop = function () {
      state.recordingBlob = new Blob(state.chunks, { type: state.recorder.mimeType || "video/webm" });
      state.uploadedFile = null;
      setCaptureStatus("录制完成", formatDuration(Date.now() - state.recordingStartedAt), "可提交");
      setBusy(false, "录制完成，可以提交评测。");
    };
    state.recorder.start();
    record.textContent = "停止录制";
    startRecordingClock();
    setBusy(false, "正在录制...");
  }

  function pickMimeType() {
    var candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    for (var i = 0; i < candidates.length; i += 1) {
      if (MediaRecorder.isTypeSupported(candidates[i])) return candidates[i];
    }
    return "";
  }

  function initSubmit() {
    var submit = $("submit-btn");
    if (!submit) return;
    submit.addEventListener("click", function () {
      if (!state.selectedLevel) {
        setBusy(false, "未找到关卡。");
        return;
      }
      var video = state.uploadedFile || state.recordingBlob;
      if (!video) {
        setBusy(false, "请先录制或选择视频。");
        setCaptureStatus("缺少视频", undefined, "无法提交");
        return;
      }
      var form = new FormData();
      form.append("levelId", state.selectedLevel.id);
      form.append("userId", "frontweb-user");
      form.append("deviceId", state.uploadedFile ? "browser-upload" : "browser-camera");
      form.append("video", video, state.uploadedFile ? state.uploadedFile.name : "challenge-recording.webm");
      setInteractionLocked(true);
      setText("keypoint-status", "识别中");
      setText("keypoint-source", "正在生成");
      startKeypointTracking();
      captureKeypointFrame();
      setCaptureStatus("提交中", undefined, "上传并识别");
      setBusy(true, "E2E 模型识别中...");
      fetch(API_BASE + "/api/e2e/challenge/submit", { method: "POST", headers: window.HUANYU_AUTH.authHeaders({}), body: form })
        .then(function (response) { return response.json(); })
        .then(function (payload) {
          if (payload.code >= 400) throw new Error(payload.message || "提交失败");
          var data = payload.data || {};
          setResult(data);
          state.progress[state.selectedLevel.id] = { score: Math.round(data.score || 0), passed: !!data.passed };
          saveProgress();
          setInteractionLocked(false);
          setCaptureStatus(data.passed ? "评测通过" : "评测完成", undefined, data.passed ? "已保存" : "可重试");
          setBusy(false, data.passed ? "通过，已保存进度。" : "未通过，可以调整动作后重试。");
        })
        .catch(function (err) {
          setInteractionLocked(false);
          setCaptureStatus("提交失败", undefined, "可重试");
          setBusy(false, "提交失败：" + err.message);
        });
    });
  }

  function boot() {
    if (window.HUANYU_AUTH && !window.HUANYU_AUTH.requireLogin()) return;
    requestJson("/api/e2e/challenge/status").then(renderStatus).catch(function () {
      renderStatus({ directE2E: { available: false }, twoStage: {} });
    });
    requestJson("/api/e2e/challenge/levels").then(function (data) {
      state.levels = data.items || [];
      renderIndexLevels();
      var params = new URLSearchParams(window.location.search);
      selectLevel(params.get("level") || (state.levels[0] && state.levels[0].id));
    }).catch(function (err) {
      setBusy(false, "关卡加载失败：" + err.message);
    });
    initFileInput();
    initCameraControls();
    initSubmit();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
