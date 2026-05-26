(function () {
  "use strict";

  var STORAGE_KEY = "huanyu-e2e-challenge-progress";

  function $(id) {
    return document.getElementById(id);
  }

  function loadChallengeProgress() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      var parsed = raw ? JSON.parse(raw) : {};
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (err) {
      return {};
    }
  }

  function toRecords(progress) {
    return Object.keys(progress || {}).map(function (levelId) {
      var item = progress[levelId] || {};
      var score = Number(item.score || 0);
      return {
        levelId: levelId,
        score: Math.max(0, Math.min(100, Math.round(score))),
        passed: Boolean(item.passed)
      };
    }).filter(function (item) {
      return item.score > 0 || item.passed;
    }).sort(function (a, b) {
      return Number(a.levelId) - Number(b.levelId);
    });
  }

  function renderStudyProgress() {
    var records = toRecords(loadChallengeProgress());
    var recent = records.length ? records[records.length - 1] : null;
    var passed = records.filter(function (item) { return item.passed; }).length;
    var average = records.length
      ? Math.round(records.reduce(function (sum, item) { return sum + item.score; }, 0) / records.length)
      : 0;

    setText("study-recent-score", recent ? recent.score + " 分" : "--");
    setText("study-recent-label", recent ? "关卡 " + recent.levelId + (recent.passed ? " 已通过" : " 待加强") : "尚未完成闯关");
    setText("study-pass-rate", passed + " / " + records.length);
    setText("study-pass-label", records.length ? "平均 " + average + " 分" : "完成后自动更新");
    setText("study-summary-label", records.length ? "已记录 " + records.length + " 关" : "本地记录");
    setText("study-next-action", nextAction(records, average));
    renderList(records);
  }

  function nextAction(records, average) {
    if (!records.length) return "先完成一次评测";
    if (average >= 85) return "继续挑战新关卡";
    if (average >= 60) return "复练低分关卡";
    return "先稳定短句动作";
  }

  function renderList(records) {
    var list = $("study-progress-list");
    if (!list) return;
    if (!records.length) {
      list.innerHTML = '<li><strong>暂无闯关记录</strong><small>完成一次 E2E 闯关评测后，这里会显示最近成绩。</small></li>';
      return;
    }
    list.innerHTML = records.map(function (item) {
      var tone = item.passed ? "is-pass" : (item.score >= 50 ? "is-warn" : "is-fail");
      var label = item.passed ? "已通过" : (item.score >= 50 ? "待加强" : "需重练");
      return [
        '<li class="' + tone + '">',
        "<strong>关卡 " + escapeHtml(item.levelId) + " · " + item.score + " 分</strong>",
        "<small>" + label + "</small>",
        "</li>"
      ].join("");
    }).join("");
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

  document.addEventListener("DOMContentLoaded", function () {
    if (window.HUANYU_AUTH && !window.HUANYU_AUTH.requireLogin()) return;
    renderStudyProgress();
  });
})();
