(function () {
  "use strict";

  var STORAGE_KEY = "huanyu-session";

  function apiBase() {
    var configured = window.HUANYU_API_BASE || "";
    configured = String(configured).trim();
    if (configured) return configured.replace(/\/+$/, "");
    var host = window.location.hostname;
    if (!host || host === "127.0.0.1" || host === "localhost") return "http://127.0.0.1:5000";
    return window.location.origin;
  }

  function readSession() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    } catch (err) {
      return null;
    }
  }

  function writeSession(session) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session || {}));
    if (session && session.user) {
      localStorage.setItem("huanyu-demo-user", JSON.stringify({
        account: session.user.username,
        identity: session.user.username === "root" ? "admin" : "user",
        signedInAt: new Date().toISOString()
      }));
    }
  }

  function clearSession() {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem("huanyu-demo-user");
  }

  function pagePrefix() {
    var pathname = window.location.pathname.replace(/\\/g, "/");
    if (pathname.indexOf("/user/13/") >= 0) return "../../";
    if (
      pathname.indexOf("/paths/") >= 0 ||
      pathname.indexOf("/developer/") >= 0 ||
      pathname.indexOf("/questions/") >= 0 ||
      pathname.indexOf("/privacy/") >= 0
    ) return "../";
    return "";
  }

  function currentRelativePath() {
    var pathname = window.location.pathname.replace(/\\/g, "/");
    var filename = pathname.split("/").pop() || "index.html";
    var next = filename;
    if (pathname.indexOf("/user/13/") >= 0) next = "user/13/" + filename;
    else if (pathname.indexOf("/paths/") >= 0) next = "paths/" + filename;
    else if (pathname.indexOf("/developer/") >= 0) next = "developer/" + filename;
    else if (pathname.indexOf("/questions/") >= 0) next = "questions/" + filename;
    else if (pathname.indexOf("/privacy/") >= 0) next = "privacy/" + filename;
    return next + window.location.search;
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (char) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[char];
    });
  }

  function authHeaders(extra) {
    var headers = extra || {};
    var session = readSession();
    if (session && session.token) headers.Authorization = "Bearer " + session.token;
    return headers;
  }

  function parseJson(response) {
    return response.text().then(function (body) {
      var payload = {};
      if (body) {
        try {
          payload = JSON.parse(body);
        } catch (err) {
          throw new Error(body || ("HTTP " + response.status));
        }
      }
      if (!response.ok || payload.code >= 400) {
        throw new Error(payload.message || ("HTTP " + response.status));
      }
      return payload;
    });
  }

  function fetchJson(path, options) {
    options = options || {};
    options.headers = authHeaders(options.headers || {});
    return fetch(apiBase() + path, options).then(parseJson);
  }

  function requireLogin() {
    var session = readSession();
    if (session && session.token) return session;
    var pathname = window.location.pathname;
    var next = pathname.split("/").pop() || "index.html";
    var prefix = "";
    if (pathname.indexOf("/paths/") >= 0) {
      prefix = "../";
      next = "paths/" + next + window.location.search;
    } else if (pathname.indexOf("/developer/") >= 0) {
      prefix = "../";
      next = "developer/" + next + window.location.search;
    } else if (pathname.indexOf("/questions/") >= 0) {
      prefix = "../";
      next = "questions/" + next + window.location.search;
    } else if (pathname.indexOf("/user/13/") >= 0) {
      prefix = "../../";
      next = "user/13/" + next + window.location.search;
    }
    window.location.href = prefix + "login.html?next=" + encodeURIComponent(next);
    return null;
  }

  function createAccountMenu(session) {
    var prefix = pagePrefix();
    var username = session && session.user && session.user.username ? session.user.username : "用户";
    var role = username === "root" ? "管理员" : "普通用户";
    var wrapper = document.createElement("div");
    wrapper.className = "nav-user-menu";
    wrapper.innerHTML = [
      '<button class="nav-user-button" type="button" aria-haspopup="true" aria-expanded="false">',
      '<i class="fa fa-user-circle" aria-hidden="true"></i>',
      '<span>个人信息管理</span>',
      '<small>' + escapeHtml(username) + '</small>',
      '</button>',
      '<div class="nav-user-dropdown" role="menu">',
      '<div class="nav-user-summary"><span>当前账号</span><strong>' + escapeHtml(username) + '</strong><small>' + role + '</small></div>',
      '<a role="menuitem" href="' + prefix + 'user/13/study.html"><i class="fa fa-line-chart" aria-hidden="true"></i> 学习与使用记录</a>',
      '<a role="menuitem" href="' + prefix + 'login.html?next=' + encodeURIComponent(currentRelativePath()) + '"><i class="fa fa-exchange" aria-hidden="true"></i> 切换账号</a>',
      '<button role="menuitem" type="button" data-huanyu-logout><i class="fa fa-sign-out" aria-hidden="true"></i> 退出登录</button>',
      '</div>'
    ].join("");

    var button = wrapper.querySelector(".nav-user-button");
    button.addEventListener("click", function (event) {
      event.stopPropagation();
      var open = wrapper.classList.toggle("is-open");
      button.setAttribute("aria-expanded", open ? "true" : "false");
    });

    wrapper.querySelector("[data-huanyu-logout]").addEventListener("click", function () {
      fetchJson("/api/auth/logout", {method: "POST"}).catch(function () {
        return null;
      }).then(function () {
        clearSession();
        window.location.href = prefix + "login.html?next=" + encodeURIComponent(currentRelativePath());
      });
    });

    document.addEventListener("click", function (event) {
      if (!wrapper.contains(event.target)) {
        wrapper.classList.remove("is-open");
        button.setAttribute("aria-expanded", "false");
      }
    });

    return wrapper;
  }

  function renderAuthNav() {
    var nav = document.querySelector(".app-nav");
    if (!nav) return;
    var oldMenu = nav.querySelector(".nav-user-menu");
    if (oldMenu) oldMenu.remove();

    var session = readSession();
    var loginAction = Array.prototype.slice.call(nav.querySelectorAll(".nav-action")).filter(function (link) {
      return (link.getAttribute("href") || "").indexOf("login.html") >= 0;
    })[0];

    if (session && session.token) {
      var menu = createAccountMenu(session);
      if (loginAction) loginAction.replaceWith(menu);
      else nav.appendChild(menu);
      return;
    }

    if (loginAction) {
      loginAction.textContent = "登录 / 进入";
      return;
    }
  }

  document.addEventListener("DOMContentLoaded", renderAuthNav);

  window.HUANYU_AUTH = {
    apiBase: apiBase,
    readSession: readSession,
    writeSession: writeSession,
    clearSession: clearSession,
    authHeaders: authHeaders,
    fetchJson: fetchJson,
    requireLogin: requireLogin,
    renderAuthNav: renderAuthNav
  };
})();
