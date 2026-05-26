(function () {
  "use strict";

  var state = {
    split: "dev",
    query: "",
    offset: 0,
    limit: 25,
    items: [],
    loading: false
  };

  function $(id) {
    return document.getElementById(id);
  }

  function apiBase() {
    var configured = window.HUANYU_API_BASE || "";
    if (configured) return configured.replace(/\/$/, "");
    if (window.location.protocol === "file:") return "http://127.0.0.1:5000";
    var host = window.location.hostname;
    var port = window.location.port;
    if ((host === "127.0.0.1" || host === "localhost") && port && port !== "5000") {
      return window.location.protocol + "//" + host + ":5000";
    }
    return window.location.origin;
  }

  function fetchJson(path) {
    return fetch(apiBase() + path, { headers: window.HUANYU_AUTH.authHeaders({}) }).then(function (response) {
      return response.json().then(function (payload) {
        if (!response.ok || payload.code >= 400) {
          throw new Error(payload.message || "请求失败");
        }
        return payload.data || {};
      });
    });
  }

  function loadSamples(reset) {
    if (state.loading) return;
    state.loading = true;
    if (reset) {
      state.offset = 0;
      state.items = [];
      renderLoading();
    }
    setLoadMore(false, "加载中...");
    var path = "/api/learning/dataset-samples?split=" + encodeURIComponent(state.split)
      + "&limit=" + state.limit
      + "&offset=" + state.offset
      + "&query=" + encodeURIComponent(state.query);

    fetchJson(path).then(function (data) {
      var items = data.items || [];
      state.items = state.items.concat(items);
      state.offset = Number(data.nextOffset || (state.offset + items.length));
      renderCourses(data);
      setLoadMore(items.length >= state.limit, "加载更多样本");
    }).catch(function (err) {
      renderError(err.message || "课程加载失败");
      setLoadMore(false, "加载失败");
    }).finally(function () {
      state.loading = false;
    });
  }

  function renderLoading() {
    var grid = $("course-grid");
    if (grid) grid.innerHTML = '<article class="course-loading">正在加载训练集课程...</article>';
  }

  function renderError(message) {
    var grid = $("course-grid");
    if (grid) {
      grid.innerHTML = '<article class="course-loading is-error">' + escapeHtml(message) + "</article>";
    }
    setText("course-source-label", "连接失败");
  }

  function renderCourses(data) {
    updateSummary(data);
    var grid = $("course-grid");
    if (!grid) return;
    if (!state.items.length) {
      grid.innerHTML = '<article class="course-loading">没有找到匹配的训练集样本。</article>';
      return;
    }
    grid.innerHTML = state.items.map(renderCourseCard).join("");
  }

  function renderCourseCard(item) {
    var title = item.chinese || "未命名句子";
    var video = item.videoUrl ? authenticatedVideoUrl(item.videoUrl) : "";
    return [
      '<article class="course-card">',
      '<div class="course-video">',
      video ? '<video src="' + escapeAttr(video) + '" controls preload="metadata" playsinline></video>' : '<div class="video-empty"><i class="fa fa-video-camera"></i><span>视频缺失</span></div>',
      "</div>",
      '<div class="course-card-body">',
      '<div class="course-card-meta"><span>' + escapeHtml(item.split) + " / " + escapeHtml(item.translator) + "</span><strong>" + escapeHtml(item.number) + "</strong></div>",
      "<h2>" + escapeHtml(title) + "</h2>",
      '<div class="course-gloss"><span>Gloss</span><p>' + escapeHtml(item.gloss || "-") + "</p></div>",
      item.note ? '<p class="course-note">' + escapeHtml(item.note) + "</p>" : "",
      '<div class="course-card-actions">',
      '<a class="button secondary" href="../paths/index.html">去闯关</a>',
      '<a class="ghost-button" href="../translate.html">上传练习</a>',
      "</div>",
      "</div>",
      "</article>"
    ].join("");
  }

  function authenticatedVideoUrl(path) {
    var url = apiBase() + path;
    var session = window.HUANYU_AUTH.readSession();
    if (!session || !session.token) return url;
    return url + (url.indexOf("?") >= 0 ? "&" : "?") + "token=" + encodeURIComponent(session.token);
  }

  function updateSummary(data) {
    setText("course-split-label", (data.split || state.split).toUpperCase());
    setText("course-count-label", String(state.items.length));
    setText("course-source-label", data.available === false ? "未找到训练集" : "CE-CSL");
  }

  function setLoadMore(enabled, text) {
    var button = $("course-load-more");
    if (!button) return;
    button.disabled = !enabled;
    button.textContent = text;
  }

  function setText(id, value) {
    var el = $(id);
    if (el) el.textContent = value;
  }

  function escapeHtml(value) {
    return String(value || "").replace(/[&<>"']/g, function (char) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char];
    });
  }

  function escapeAttr(value) {
    return escapeHtml(value).replace(/`/g, "&#096;");
  }

  function bindEvents() {
    Array.prototype.forEach.call(document.querySelectorAll("[data-split]"), function (button) {
      button.addEventListener("click", function () {
        state.split = button.getAttribute("data-split") || "dev";
        Array.prototype.forEach.call(document.querySelectorAll("[data-split]"), function (item) {
          item.classList.toggle("active", item === button);
        });
        loadSamples(true);
      });
    });

    var form = $("course-search-form");
    if (form) {
      form.addEventListener("submit", function (event) {
        event.preventDefault();
        state.query = ($("course-search-input") || {}).value || "";
        loadSamples(true);
      });
    }

    var more = $("course-load-more");
    if (more) {
      more.addEventListener("click", function () {
        loadSamples(false);
      });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (!window.HUANYU_AUTH.requireLogin()) return;
    bindEvents();
    loadSamples(true);
  });
})();
