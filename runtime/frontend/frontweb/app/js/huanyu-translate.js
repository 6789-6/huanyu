(function () {
    if (window.HUANYU_AUTH && !window.HUANYU_AUTH.requireLogin()) return;
    var input = document.getElementById("video-input");
    var preview = document.getElementById("preview-video");
    var empty = document.getElementById("video-empty");
    var fileName = document.getElementById("file-name");
    var translateButton = document.getElementById("translate-button");
    var demoButton = document.getElementById("demo-button");
    var state = document.getElementById("pipeline-state");
    var translation = document.getElementById("translation-output");
    var gloss = document.getElementById("gloss-output");
    var confidence = document.getElementById("confidence-output");
    var latency = document.getElementById("latency-output");
    var service = document.getElementById("service-output");
    var inputStatus = document.getElementById("input-status");
    var uploadProgress = document.getElementById("upload-progress");
    var serviceState = document.getElementById("service-state");
    var resultServiceState = document.getElementById("result-service-state");
    var qualityRow = document.getElementById("quality-row");
    var quality = document.getElementById("quality-output");
    var rawTextBlock = document.getElementById("raw-text-block");
    var rawText = document.getElementById("raw-text-output");
    var referenceRow = document.getElementById("reference-row");
    var referenceOutput = document.getElementById("reference-output");
    var referenceGlossOutput = document.getElementById("reference-gloss-output");
    var demoSampleList = document.getElementById("demo-sample-list");
    var demoSampleState = document.getElementById("demo-sample-state");
    var historyList = document.getElementById("history-list");
    var deviceState = document.getElementById("device-state");
    var alertButton = document.getElementById("guardian-alert-button");
    var recommendationList = document.getElementById("recommendation-list");
    var dialoguePanel = document.getElementById("dialogue-panel");
    var analysisOutput = document.getElementById("analysis-output");
    var emotionOutput = document.getElementById("emotion-output");
    var analyzeDialogueButton = document.getElementById("analyze-dialogue-button");
    var voiceSelect = document.getElementById("voice-select");
    var ttsButton = document.getElementById("tts-button");
    var ttsAudio = document.getElementById("tts-audio");
    var ttsState = document.getElementById("tts-state");
    var realtimeStartButton = document.getElementById("realtime-start-button");
    var realtimeStopButton = document.getElementById("realtime-stop-button");
    var realtimeStatus = document.getElementById("realtime-status");
    var realtimeKeywords = document.getElementById("realtime-keywords");
    var realtimeSentence = document.getElementById("realtime-sentence");
    var realtimeModel = document.getElementById("realtime-model");
    var shell = document.querySelector(".translate-shell");
    var capturePanel = document.querySelector(".capture-panel");
    var resultPanel = document.querySelector(".result-panel");
    var API_BASE = getApiBase();
    var busy = false;
    var lastResultText = "";
    var lastResultGloss = "";
    var realtimeStream = null;
    var realtimeTimer = null;
    var realtimeInFlight = false;
    var realtimeCanvas = document.createElement("canvas");

    var demo = {
        text: "我写作业需要一些帮助。",
        gloss: "我/写字/作业/帮助（我）/需要2/。",
        confidence: "91%",
        latency: "1.8s",
        service: "CorrNet + G2T",
        showQuality: false
    };

    function setTranslateStatus(inputText, progressText, serviceText) {
        if (inputStatus) {
            inputStatus.textContent = inputText || "等待视频";
        }
        if (uploadProgress) {
            uploadProgress.textContent = progressText || "待提交";
        }
        if (serviceState) {
            serviceState.textContent = serviceText || "未检测";
        }
        if (resultServiceState) {
            resultServiceState.textContent = serviceText || "可请求";
        }
    }

    function resetResultTone() {
        if (resultPanel) {
            resultPanel.classList.remove("is-success", "is-warning", "is-error");
        }
        if (capturePanel) {
            capturePanel.classList.remove("is-ready", "is-error");
        }
    }

    function setResultTone(tone) {
        resetResultTone();
        if (resultPanel && tone) {
            resultPanel.classList.add("is-" + tone);
        }
        if (capturePanel && tone === "success") {
            capturePanel.classList.add("is-ready");
        }
        if (capturePanel && tone === "error") {
            capturePanel.classList.add("is-error");
        }
    }

    function setResult(data, label) {
        state.textContent = label || "已完成";
        translation.textContent = data.text;
        gloss.textContent = data.gloss;
        confidence.textContent = data.confidence;
        latency.textContent = data.latency;
        service.textContent = data.service;
        if (qualityRow) {
            qualityRow.hidden = !data.showQuality;
        }
        if (quality) {
            quality.textContent = data.quality || "-";
        }
        if (rawTextBlock) {
            rawTextBlock.style.display = data.rawText ? "block" : "none";
        }
        if (rawText) {
            rawText.textContent = data.rawText || "-";
        }
        if (referenceRow) {
            referenceRow.hidden = !data.referenceChinese;
        }
        if (referenceOutput) {
            referenceOutput.textContent = data.referenceChinese || "-";
        }
        if (referenceGlossOutput) {
            referenceGlossOutput.textContent = data.referenceGloss || "-";
        }
        lastResultText = (data.text || "").trim();
        lastResultGloss = data.gloss || "";
        if (ttsState) {
            ttsState.textContent = lastResultText ? "可播报当前结果" : "等待翻译结果";
        }
        setResultTone(data.showQuality ? "warning" : "success");
        setTranslateStatus("结果已生成", "已完成", data.service || "可请求");
        updateActionAvailability();
    }

    function setBusy(nextBusy) {
        busy = Boolean(nextBusy);
        if (shell) {
            shell.classList.toggle("is-busy", busy);
            shell.setAttribute("aria-busy", busy ? "true" : "false");
        }
        [translateButton, demoButton, alertButton].forEach(function (button) {
            if (button) {
                button.disabled = busy;
            }
        });
        if (demoSampleList) {
            Array.prototype.forEach.call(demoSampleList.querySelectorAll("button"), function (button) {
                button.disabled = busy || button.hasAttribute("data-unavailable");
            });
        }
        updateActionAvailability();
    }

    function escapeHtml(value) {
        return String(value || "").replace(/[&<>"']/g, function (ch) {
            return {
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                "\"": "&quot;",
                "'": "&#39;"
            }[ch];
        });
    }

    function getCurrentResultText() {
        return (lastResultText || "").trim();
    }

    function updateActionAvailability() {
        var hasText = Boolean(getCurrentResultText());
        var hasVoice = Boolean(voiceSelect && voiceSelect.value);
        if (ttsButton) {
            ttsButton.disabled = busy || !hasText || !hasVoice;
        }
        if (analyzeDialogueButton) {
            analyzeDialogueButton.disabled = busy || !hasText;
        }
    }

    function clearDialogueOutputs(message) {
        lastResultText = "";
        lastResultGloss = "";
        if (emotionOutput) {
            emotionOutput.textContent = "待分析";
        }
        if (analysisOutput) {
            analysisOutput.textContent = message || "等待新的翻译结果。";
        }
        if (recommendationList) {
            recommendationList.innerHTML = "<li><strong>等待推荐回复</strong><small>翻译完成后自动生成。</small></li>";
        }
        if (ttsState) {
            ttsState.textContent = "等待翻译结果";
        }
        if (ttsAudio) {
            if (ttsAudio.getAttribute("data-audio-url")) {
                URL.revokeObjectURL(ttsAudio.getAttribute("data-audio-url"));
                ttsAudio.removeAttribute("data-audio-url");
            }
            ttsAudio.hidden = true;
            ttsAudio.removeAttribute("src");
        }
        updateActionAvailability();
    }

    function isLocalDevHost() {
        var host = window.location && window.location.hostname;
        return host === "127.0.0.1" || host === "localhost";
    }

    function getApiBase() {
        var configured = window.HUANYU_API_BASE || "";
        configured = String(configured).trim();
        if (configured) {
            return configured.replace(/\/+$/, "");
        }
        if (window.location && /^https?:$/.test(window.location.protocol) && window.location.port === "5000") {
            return window.location.origin;
        }
        if (isLocalDevHost()) {
            return "http://127.0.0.1:5000";
        }
        return window.location.origin;
    }

    function parseJsonResponse(response) {
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
        options.headers = window.HUANYU_AUTH ? window.HUANYU_AUTH.authHeaders(options.headers || {}) : (options.headers || {});
        return fetch(API_BASE + path, options || {}).then(parseJsonResponse);
    }

    function postJson(path, payload) {
        return fetchJson(path, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload || {})
        });
    }

    function formatConfidence(value) {
        if (typeof value !== "number") {
            return "--";
        }
        return Math.round(value * 100) + "%";
    }

    function formatFallbackReason(reason) {
        var labels = {
            short_gloss: "短 gloss，显示保守可读结果",
            low_vocab_hit_rate: "词表命中低，显示 gloss 可读结果",
            empty_translation: "翻译为空，显示 gloss 可读结果",
            repeated_source_term: "检测到重复扩写，显示保守结果",
            over_expanded: "检测到过度扩写，显示保守结果",
            pipeline_failed: "模型失败，使用演示降级结果",
            pipeline_import_error: "算法服务未加载，使用演示降级结果"
        };
        return labels[reason] || (reason ? ("降级原因：" + reason) : "直接使用 G2T 输出");
    }

    function formatEmotion(emotion) {
        var labels = {
            positive: "积极",
            neutral: "中性",
            negative: "消极"
        };
        return labels[emotion] || "中性";
    }

    function formatEmotionScores(scores) {
        if (!scores) {
            return "";
        }
        return "积极 " + formatConfidence(scores.positive) + " / 中性 " +
            formatConfidence(scores.neutral) + " / 消极 " + formatConfidence(scores.negative);
    }

    function renderEmotionSummary(emotion, scores) {
        if (emotionOutput) {
            emotionOutput.textContent = formatEmotion(emotion);
        }
        if (analysisOutput) {
            analysisOutput.textContent = formatEmotionScores(scores) || "当前对话可继续交流。";
        }
    }

    function renderRecommendations(items, source) {
        if (!recommendationList) {
            return;
        }
        if (!items || !items.length) {
            recommendationList.innerHTML = "<li><strong>暂无推荐回复</strong><small>等待新的识别结果。</small></li>";
            return;
        }
        recommendationList.innerHTML = items.map(function (item) {
            var text = String(item || "");
            return "<li class=\"recommendation-item\"><div><strong>" + escapeHtml(text) +
                "</strong><small>" + escapeHtml(source || "可用于快速语音播报。") +
                "</small></div><button type=\"button\" data-recommendation-text=\"" +
                escapeHtml(text) + "\"><i class=\"fa fa-volume-up\" aria-hidden=\"true\"></i><span>播报</span></button></li>";
        }).join("");
    }

    function analyzeEmotion(text) {
        var message = (text || getCurrentResultText()).trim();
        if (!message) {
            return Promise.resolve(null);
        }
        return postJson("/api/dialogue/emotion", {text: message})
            .then(function (payload) {
                var data = payload.data || {};
                if (emotionOutput) {
                    emotionOutput.textContent = formatEmotion(data.emotion) + " / " + formatConfidence(data.confidence);
                }
                return data;
            })
            .catch(function (err) {
                if (emotionOutput) {
                    emotionOutput.textContent = "情绪不可用";
                }
                if (analysisOutput) {
                    analysisOutput.textContent = "情绪接口暂不可用：" + err.message;
                }
                return null;
            });
    }

    function analyzeCurrentDialogue() {
        var message = getCurrentResultText();
        if (!message) {
            if (analysisOutput) {
                analysisOutput.textContent = "暂无可分析的翻译结果。";
            }
            return Promise.resolve(null);
        }
        if (dialoguePanel) {
            dialoguePanel.classList.add("is-loading");
        }
        if (emotionOutput) {
            emotionOutput.textContent = "分析中";
        }
        if (analysisOutput) {
            analysisOutput.textContent = "正在分析当前翻译结果。";
        }
        return postJson("/api/dialogue/analyze", {
            conversationId: "browser-demo",
            messages: [{
                role: "user",
                content: message,
                time: new Date().toISOString()
            }]
        }).then(function (payload) {
            var data = payload.data || {};
            renderEmotionSummary(data.emotion, data.emotionScores);
            renderRecommendations(data.recommendations, "对话分析推荐。");
            return analyzeEmotion(message);
        }).catch(function (err) {
            if (emotionOutput) {
                emotionOutput.textContent = "分析失败";
            }
            if (analysisOutput) {
                analysisOutput.textContent = err.message;
            }
            return null;
        }).then(function (result) {
            if (dialoguePanel) {
                dialoguePanel.classList.remove("is-loading");
            }
            updateActionAvailability();
            return result;
        });
    }

    function recommendReplies(text) {
        var message = (text || getCurrentResultText()).trim();
        if (!recommendationList) {
            return Promise.resolve(null);
        }
        if (!message) {
            renderRecommendations([], "");
            return Promise.resolve(null);
        }
        recommendationList.innerHTML = "<li><strong>正在生成推荐回复</strong><small>等待对话接口返回。</small></li>";
        return postJson("/api/dialogue/recommend", {
            lastMessage: message,
            history: lastResultGloss ? [lastResultGloss] : []
        }).then(function (payload) {
            var data = payload.data || {};
            renderRecommendations(data.recommendations || [], "推荐回复接口。");
            return data;
        }).catch(function (err) {
            recommendationList.innerHTML = "<li><strong>推荐暂不可用</strong><small>" +
                escapeHtml(err.message) + "</small></li>";
            return null;
        });
    }

    function refreshDialogueTools() {
        var message = getCurrentResultText();
        if (!message) {
            return Promise.resolve(null);
        }
        return analyzeCurrentDialogue().then(function () {
            return recommendReplies(message);
        });
    }

    function loadVoices() {
        if (!voiceSelect) {
            return Promise.resolve(null);
        }
        voiceSelect.disabled = true;
        return fetchJson("/api/tts/voices")
            .then(function (payload) {
                var voices = (payload.data && payload.data.voices) || [];
                if (!voices.length) {
                    throw new Error("暂无可用音色");
                }
                voiceSelect.innerHTML = voices.map(function (voice) {
                    return "<option value=\"" + escapeHtml(voice.id) + "\">" +
                        escapeHtml(voice.name) + "</option>";
                }).join("");
                voiceSelect.value = voices.some(function (voice) {
                    return voice.id === "zh-CN-XiaoxiaoNeural";
                }) ? "zh-CN-XiaoxiaoNeural" : voices[0].id;
                voiceSelect.disabled = false;
                if (ttsState) {
                    ttsState.textContent = getCurrentResultText() ? "可播报当前结果" : "等待翻译结果";
                }
                updateActionAvailability();
                return voices;
            })
            .catch(function (err) {
                voiceSelect.innerHTML = "<option value=\"\">音色不可用</option>";
                voiceSelect.disabled = true;
                if (ttsState) {
                    ttsState.textContent = "语音服务不可用：" + err.message;
                }
                updateActionAvailability();
                return null;
            });
    }

    function synthesizeSpeech(textOverride) {
        var text = (textOverride || getCurrentResultText()).trim();
        if (!text) {
            if (ttsState) {
                ttsState.textContent = "暂无可播报内容";
            }
            return Promise.resolve(null);
        }
        if (!voiceSelect || !voiceSelect.value) {
            if (ttsState) {
                ttsState.textContent = "音色未加载完成";
            }
            return Promise.resolve(null);
        }
        setBusy(true);
        if (ttsState) {
            ttsState.textContent = "正在合成语音。";
        }
        return fetch(API_BASE + "/api/tts/synthesize", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                text: text,
                voiceType: voiceSelect.value,
                speed: 1.0,
                pitch: 0
            })
        }).then(function (response) {
            if (!response.ok) {
                return response.text().then(function (body) {
                    var message = body;
                    try {
                        message = JSON.parse(body).message || body;
                    } catch (ignored) {
                        message = body;
                    }
                    throw new Error(message || ("HTTP " + response.status));
                });
            }
            return response.blob();
        }).then(function (blob) {
            var audioUrl = URL.createObjectURL(blob);
            if (ttsAudio) {
                if (ttsAudio.getAttribute("data-audio-url")) {
                    URL.revokeObjectURL(ttsAudio.getAttribute("data-audio-url"));
                }
                ttsAudio.src = audioUrl;
                ttsAudio.setAttribute("data-audio-url", audioUrl);
                ttsAudio.hidden = false;
                var playPromise = ttsAudio.play();
                if (playPromise && playPromise.catch) {
                    playPromise.catch(function () {});
                }
            }
            if (ttsState) {
                ttsState.textContent = "语音已生成";
            }
            return blob;
        }).catch(function (err) {
            if (ttsState) {
                ttsState.textContent = "语音合成失败：" + err.message;
            }
            return null;
        }).then(function (result) {
            setBusy(false);
            return result;
        });
    }

    function renderApiResult(payload) {
        var data = payload.data || {};
        var serviceName = data.model || "CorrNet + G2T";
        if (data.fallback) {
            serviceName += " / 保守输出";
        }
        setResult({
            text: data.result || "未返回中文翻译",
            gloss: data.gloss || "-",
            confidence: formatConfidence(data.confidence),
            latency: data.latencyMs ? (data.latencyMs / 1000).toFixed(1) + "s" : "--",
            service: serviceName,
            quality: formatFallbackReason(data.fallbackReason),
            rawText: data.rawText || "",
            referenceChinese: data.referenceChinese || "",
            referenceGloss: data.referenceGloss || "",
            showQuality: Boolean(data.fallback || data.rawText || data.fallbackReason)
        }, data.status === "completed" ? "真实结果" : "演示/降级结果");
    }

    function renderDemoSamples(items) {
        if (!demoSampleList) {
            return;
        }
        if (!items.length) {
            demoSampleList.innerHTML = "<li><span>暂无精选样本</span></li>";
            return;
        }
        demoSampleList.innerHTML = items.map(function (item) {
            var disabled = item.available ? "" : " disabled";
            var unavailable = item.available ? "" : " data-unavailable=\"true\"";
            return "<li><div><strong>" + escapeHtml(item.title) + "</strong><small>" +
                escapeHtml(item.referenceChinese) + "</small></div><button type=\"button\" data-sample-id=\"" +
                escapeHtml(item.id) + "\"" + unavailable + disabled + ">运行</button></li>";
        }).join("");
        setBusy(busy);
    }

    function loadDemoSamples() {
        if (!demoSampleList) {
            return Promise.resolve();
        }
        return fetchJson("/api/recognition/demo-samples")
            .then(function (payload) {
                var items = (payload.data && payload.data.items) || [];
                if (demoSampleState) {
                    demoSampleState.textContent = items.length + " 个";
                }
                renderDemoSamples(items);
            })
            .catch(function (err) {
                if (demoSampleState) {
                    demoSampleState.textContent = "不可用";
                }
                demoSampleList.innerHTML = "<li class=\"sample-offline\"><div><strong>样本服务不可用</strong><small>" +
                    escapeHtml(err.message) + "</small></div><button type=\"button\" data-unavailable=\"true\" disabled>离线</button></li>";
                setTranslateStatus("服务离线", "样本不可用", "检查 Flask");
            });
    }

    function runDemoSample(sampleId) {
        if (busy) {
            return Promise.resolve();
        }
        setBusy(true);
        state.textContent = "运行精选样本";
        translation.textContent = "模型正在处理本地演示视频。";
        gloss.textContent = "处理中";
        confidence.textContent = "--";
        latency.textContent = "--";
        service.textContent = "Flask API";
        clearDialogueOutputs("样本运行中，等待翻译结果。");
        if (qualityRow) {
            qualityRow.hidden = true;
        }
        if (referenceRow) {
            referenceRow.hidden = true;
        }

        return fetchJson("/api/recognition/demo-samples/" + encodeURIComponent(sampleId) + "/run", {
            method: "POST",
            body: new FormData()
        }).then(function (payload) {
            fileName.textContent = "精选样本：" + (payload.data && payload.data.title ? payload.data.title : sampleId);
            if (preview) {
                preview.removeAttribute("src");
                preview.style.display = "none";
            }
            empty.style.display = "flex";
            renderApiResult(payload);
            return refreshDialogueTools().then(refreshHistory);
        }).catch(function (err) {
            state.textContent = "样本失败";
            translation.textContent = "无法运行精选样本：" + err.message;
            gloss.textContent = "-";
            confidence.textContent = "--";
            latency.textContent = "--";
            service.textContent = "检查 Flask 是否启动";
            setResultTone("error");
            setTranslateStatus("接口失败", "样本可重试", "服务离线");
        }).then(function () {
            setBusy(false);
        });
    }

    function uploadVideo(file) {
        var form = new FormData();
        form.append("video", file);
        form.append("userId", "web-user");
        form.append("deviceId", "browser-demo");
        return fetchJson("/api/recognition/upload", {
            method: "POST",
            body: form
        });
    }

    function refreshHistory() {
        if (!historyList) {
            return Promise.resolve();
        }
        return fetchJson("/api/conversation/history?limit=8")
            .then(function (payload) {
                var items = (payload.data && payload.data.items) || [];
                if (!items.length) {
                    historyList.innerHTML = "<li><strong>暂无记录</strong><small>完成一次翻译后会显示在这里。</small></li>";
                    return;
                }
                historyList.innerHTML = items.map(function (item) {
                    return "<li><span>" + escapeHtml((item.time || "").slice(11, 19)) + "</span><strong>" +
                        escapeHtml(item.content || "") + "</strong><small>" + escapeHtml(item.gloss || item.deviceId || "") + "</small></li>";
                }).join("");
            })
            .catch(function (err) {
                historyList.innerHTML = "<li><strong>记录暂不可用</strong><small>" + escapeHtml(err.message) + "</small></li>";
            });
    }

    function refreshDevice() {
        if (!deviceState) {
            return Promise.resolve();
        }
        return fetchJson("/api/device/status")
            .then(function (payload) {
                var data = payload.data || {};
                deviceState.textContent = (data.online ? "在线" : "离线") + " / 摄像头：" + (data.camera || "unknown");
            })
            .catch(function () {
                deviceState.textContent = "设备状态不可用";
            });
    }

    function setRealtimeStatus(text, modelText) {
        if (realtimeStatus) {
            realtimeStatus.textContent = text || "待启动";
        }
        if (realtimeModel && modelText) {
            realtimeModel.textContent = modelText;
        }
    }

    function renderRealtimeResult(payload) {
        var data = payload.data || {};
        var words = (data.keywords || []).map(function (item) {
            return item.token + (item.confidence ? " " + Math.round(item.confidence * 100) + "%" : "");
        });
        if (realtimeKeywords) {
            realtimeKeywords.textContent = words.length ? words.join(" / ") : "--";
        }
        if (realtimeSentence) {
            realtimeSentence.textContent = data.sentence || "正在观察动作...";
        }
        setRealtimeStatus(data.fallback ? "实时模型离线" : "识别中", data.model || "keyword realtime");
        if (data.sentence && !data.fallback) {
            translation.textContent = data.sentence;
            gloss.textContent = data.gloss || words.join("/");
            confidence.textContent = data.confidence ? Math.round(data.confidence * 100) + "%" : "--";
            latency.textContent = data.latencyMs ? data.latencyMs + "ms" : "--";
            service.textContent = data.model || "keyword realtime";
            lastResultText = data.sentence;
            lastResultGloss = data.gloss || "";
            updateActionAvailability();
        }
    }

    function sendRealtimeFrame() {
        if (!realtimeStream || !preview || realtimeInFlight || !preview.videoWidth) {
            return;
        }
        realtimeInFlight = true;
        realtimeCanvas.width = 320;
        realtimeCanvas.height = Math.max(180, Math.round(320 * (preview.videoHeight || 180) / (preview.videoWidth || 320)));
        realtimeCanvas.getContext("2d").drawImage(preview, 0, 0, realtimeCanvas.width, realtimeCanvas.height);
        fetchJson("/api/recognition/realtime/frame", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                frameData: realtimeCanvas.toDataURL("image/jpeg", 0.72),
                deviceId: "browser-realtime",
                timestamp: Date.now()
            })
        }).then(renderRealtimeResult).catch(function (err) {
            setRealtimeStatus("接口离线", "fallback: " + err.message);
            if (realtimeSentence) {
                realtimeSentence.textContent = "实时接口暂不可用，可继续使用上传视频识别。";
            }
        }).then(function () {
            realtimeInFlight = false;
        });
    }

    function isCameraContextAllowed() {
        return window.isSecureContext || isLocalDevHost();
    }

    function prepareRealtimePreview(stream) {
        preview.pause();
        preview.srcObject = null;
        preview.removeAttribute("src");
        preview.load();
        preview.srcObject = stream;
        preview.controls = false;
        preview.style.display = "block";
        empty.style.display = "none";
        return preview.play();
    }

    function startRealtimeRecognition() {
        if (!isCameraContextAllowed()) {
            setRealtimeStatus("请用 localhost 打开页面", "摄像头需要 http://127.0.0.1:8080 或 HTTPS");
            return;
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setRealtimeStatus("浏览器不支持摄像头");
            return;
        }
        stopRealtimeRecognition(false);
        setRealtimeStatus("正在打开摄像头", "keyword realtime");
        fetchJson("/api/recognition/realtime/status").then(function (payload) {
            var data = payload.data || {};
            if (realtimeModel) {
                realtimeModel.textContent = data.available ? data.model : "fallback: " + (data.error || "模型未就绪");
            }
        }).catch(function () {
            setRealtimeStatus("状态接口离线");
        });
        navigator.mediaDevices.getUserMedia({video: true, audio: false}).then(function (stream) {
            realtimeStream = stream;
            return prepareRealtimePreview(stream);
        }).then(function () {
            fileName.textContent = "实时摄像头输入";
            state.textContent = "实时识别中";
            setTranslateStatus("实时摄像头", "持续采样", "keyword realtime");
            setRealtimeStatus("识别中", "keyword realtime");
            if (realtimeStartButton) realtimeStartButton.disabled = true;
            if (realtimeStopButton) realtimeStopButton.disabled = false;
            realtimeTimer = window.setInterval(sendRealtimeFrame, 1000);
            sendRealtimeFrame();
        }).catch(function (err) {
            setRealtimeStatus("摄像头不可用", err.message);
            stopRealtimeRecognition(false);
        });
    }

    function stopRealtimeRecognition(updateStatus) {
        if (realtimeTimer) {
            window.clearInterval(realtimeTimer);
            realtimeTimer = null;
        }
        if (realtimeStream) {
            realtimeStream.getTracks().forEach(function (track) { track.stop(); });
            realtimeStream = null;
        }
        realtimeInFlight = false;
        if (preview && preview.srcObject) {
            preview.pause();
            preview.srcObject = null;
            preview.controls = true;
        }
        if (realtimeStartButton) realtimeStartButton.disabled = false;
        if (realtimeStopButton) realtimeStopButton.disabled = true;
        if (updateStatus !== false) {
            setRealtimeStatus("已停止");
        }
    }

    input.addEventListener("change", function () {
        if (busy) {
            return;
        }
        stopRealtimeRecognition(false);
        var file = input.files && input.files[0];
        if (!file) {
            return;
        }
        fileName.textContent = file.name;
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
        empty.style.display = "none";
        state.textContent = "视频已载入";
        translation.textContent = "点击“开始翻译”提交到算法服务。";
        gloss.textContent = "-";
        confidence.textContent = "--";
        latency.textContent = "--";
        service.textContent = "等待接口";
        setResultTone("success");
        setTranslateStatus("已选择视频", "待提交", "等待接口");
        clearDialogueOutputs("视频已载入，等待翻译。");
        if (qualityRow) {
            qualityRow.hidden = true;
        }
        if (referenceRow) {
            referenceRow.hidden = true;
        }
    });

    demoButton.addEventListener("click", function () {
        if (busy) {
            return;
        }
        fileName.textContent = "演示样例：test-00328.mp4";
        if (preview) {
            preview.removeAttribute("src");
            preview.style.display = "none";
        }
        empty.style.display = "flex";
        setTranslateStatus("演示样例", "本地演示", "无需接口");
        setResult(demo, "演示结果");
        refreshDialogueTools();
    });

    translateButton.addEventListener("click", function () {
        if (busy) {
            return;
        }
        if (!(input.files && input.files[0])) {
            setTranslateStatus("未选择视频", "使用演示", "无需接口");
            setResult(demo, "演示结果");
            refreshDialogueTools();
            return;
        }

        setBusy(true);
        resetResultTone();
        if (capturePanel) {
            capturePanel.classList.add("is-ready");
        }
        state.textContent = "正在上传";
        translation.textContent = "视频已提交，等待算法服务返回。";
        gloss.textContent = "处理中";
        confidence.textContent = "--";
        latency.textContent = "--";
        service.textContent = "Flask API";
        setTranslateStatus("已提交视频", "上传并识别", "请求中");
        clearDialogueOutputs("视频已提交，等待翻译结果。");
        if (qualityRow) {
            qualityRow.hidden = true;
        }
        if (referenceRow) {
            referenceRow.hidden = true;
        }

        uploadVideo(input.files[0])
            .then(function (result) {
                renderApiResult(result);
                return refreshDialogueTools().then(refreshHistory);
            })
            .catch(function (err) {
                state.textContent = "接口失败";
                translation.textContent = "无法连接算法服务：" + err.message;
                gloss.textContent = "-";
                confidence.textContent = "--";
                latency.textContent = "--";
                service.textContent = "检查 Flask 是否启动";
                setResultTone("error");
                setTranslateStatus("接口失败", "可重试", "服务离线");
                if (qualityRow) {
                    qualityRow.hidden = true;
                }
                if (referenceRow) {
                    referenceRow.hidden = true;
                }
            })
            .then(function () {
                setBusy(false);
            });
    });

    if (demoSampleList) {
        demoSampleList.addEventListener("click", function (event) {
            var target = event.target;
            if (!target || !target.getAttribute) {
                return;
            }
            var sampleId = target.getAttribute("data-sample-id");
            if (sampleId) {
                runDemoSample(sampleId);
            }
        });
    }

    if (alertButton) {
        alertButton.addEventListener("click", function () {
            fetchJson("/api/emergency/alert", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({deviceId: "browser-demo"})
            }).then(function () {
                return refreshHistory();
            });
        });
    }

    if (voiceSelect) {
        voiceSelect.addEventListener("change", updateActionAvailability);
    }

    if (ttsButton) {
        ttsButton.addEventListener("click", function () {
            if (!busy) {
                synthesizeSpeech();
            }
        });
    }

    if (analyzeDialogueButton) {
        analyzeDialogueButton.addEventListener("click", function () {
            if (!busy) {
                refreshDialogueTools();
            }
        });
    }

    if (recommendationList) {
        recommendationList.addEventListener("click", function (event) {
            var target = event.target;
            var button = target && target.closest ? target.closest("[data-recommendation-text]") : null;
            if (button && !busy) {
                synthesizeSpeech(button.getAttribute("data-recommendation-text") || "");
            }
        });
    }

    if (realtimeStartButton) {
        realtimeStartButton.addEventListener("click", startRealtimeRecognition);
    }

    if (realtimeStopButton) {
        realtimeStopButton.addEventListener("click", function () {
            stopRealtimeRecognition(true);
        });
    }

    refreshHistory();
    refreshDevice();
    loadDemoSamples();
    loadVoices();
    updateActionAvailability();
    setInterval(refreshDevice, 10000);
}());
