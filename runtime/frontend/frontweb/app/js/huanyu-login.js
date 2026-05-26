(function () {
  "use strict";

  function $(id) {
    return document.getElementById(id);
  }

  function setMessage(text, tone) {
    var message = $("login-message");
    if (!message) return;
    message.textContent = text;
    message.classList.remove("is-error", "is-success");
    if (tone) message.classList.add(tone);
  }

  function credentials() {
    var account = $("login-account");
    var password = $("login-password");
    return {
      username: account ? account.value.trim() : "",
      password: password ? password.value : ""
    };
  }

  function nextUrl() {
    var next = new URLSearchParams(window.location.search).get("next");
    return next || "index.html";
  }

  function submit(path, label) {
    var data = credentials();
    if (!data.username) {
      setMessage("请输入用户名。", "is-error");
      $("login-account").focus();
      return;
    }
    if (!data.password) {
      setMessage("请输入密码。", "is-error");
      $("login-password").focus();
      return;
    }
    setMessage(label + "中...", "");
    fetch(window.HUANYU_AUTH.apiBase() + path, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    }).then(function (response) {
      return response.json().then(function (payload) {
        if (!response.ok || payload.code >= 400) {
          throw new Error(payload.message || label + "失败");
        }
        return payload.data || {};
      });
    }).then(function (payload) {
      window.HUANYU_AUTH.writeSession(payload);
      setMessage(label + "成功，正在进入。", "is-success");
      window.setTimeout(function () {
        window.location.href = nextUrl();
      }, 220);
    }).catch(function (err) {
      setMessage(err.message || label + "失败", "is-error");
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var form = $("login-form");
    var register = $("login-register-button");
    if (form) {
      form.addEventListener("submit", function (event) {
        event.preventDefault();
        submit("/api/auth/login", "登录");
      });
    }
    if (register) {
      register.addEventListener("click", function () {
        submit("/api/auth/register", "注册");
      });
    }
  });
})();
