#!/usr/bin/env python3
# ========================================================
# 唤语 (HuanYu) 算法服务
# Python Flask API - 手语识别 + 对话分析 + TTS
# ========================================================

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import csv
import hashlib
import hmac
import os
import secrets
import sys
import json
import uuid
import asyncio
import threading
import re
import math
from datetime import datetime
import tempfile
import time
from pathlib import Path

app = Flask(__name__)
CORS(app)

APP_ROOT = Path(__file__).resolve().parent
PORTABLE_ROOT = APP_ROOT.parent.parent
PACKAGE_ROOT = PORTABLE_ROOT.parent
CE_CSL_ROOT = Path(os.environ.get("CE_CSL_ROOT", "data/CE-CSL")).resolve()

# 上传文件保存路径
UPLOAD_FOLDER = str(APP_ROOT / 'uploads')
RESULT_FOLDER = str(APP_ROOT / 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(APP_ROOT / 'data', exist_ok=True)
COMMUNITY_DATA_FILE = APP_ROOT / 'data' / 'community.json'
USERS_DATA_FILE = APP_ROOT / 'data' / 'users.json'
SESSIONS_DATA_FILE = APP_ROOT / 'data' / 'sessions.json'
USER_USAGE_DIR = APP_ROOT / 'data' / 'user_usage'
os.makedirs(USER_USAGE_DIR, exist_ok=True)
UPLOAD_FOLDER_ABS = os.path.abspath(UPLOAD_FOLDER)
RESULT_FOLDER_ABS = os.path.abspath(RESULT_FOLDER)

# 模拟任务状态存储
tasks = {}
conversation_history = []
device_status = {
    'deviceId': 'raspi-demo-001',
    'online': False,
    'camera': 'unknown',
    'battery': None,
    'lastSeen': None,
}

DEMO_SAMPLES = [
    {
        'id': 'dev-00367',
        'title': '住房场景',
        'split': 'dev',
        'translator': 'H',
        'videoPath': str(PORTABLE_ROOT / 'demo_assets' / 'videos' / 'dev-00367.mp4'),
        'referenceChinese': '房子还没弄好。',
        'referenceGloss': '房/建设/好/没/。',
        'note': '真实模型可识别到住房相关关键词，适合展示保守输出。',
    },
    {
        'id': 'dev-00422',
        'title': '询问位置',
        'split': 'dev',
        'translator': 'K',
        'videoPath': str(PORTABLE_ROOT / 'demo_assets' / 'videos' / 'dev-00422.mp4'),
        'referenceChinese': '请问卧室在哪里？',
        'referenceGloss': '卧室/在/哪/？',
        'note': '短 gloss 场景，适合展示质量闸门。',
    },
    {
        'id': 'dev-00142',
        'title': '帮助请求',
        'split': 'dev',
        'translator': 'C',
        'videoPath': str(PORTABLE_ROOT / 'demo_assets' / 'videos' / 'dev-00142.mp4'),
        'referenceChinese': '你需要帮忙吗？',
        'referenceGloss': '你/帮助/要/？',
        'note': '问句意图场景，适合展示真实模型局限。',
    },
]
DEMO_SAMPLE_BY_ID = {sample['id']: sample for sample in DEMO_SAMPLES}

E2E_CHALLENGE_LEVELS = [
    {
        'id': '1',
        'title': '基础问候',
        'targetChinese': '你好。',
        'targetGloss': '你/好/。',
        'passScore': 60,
        'tips': '手型清晰，动作停顿稳定。',
    },
    {
        'id': '2',
        'title': '请求帮助',
        'targetChinese': '我写作业需要一些帮助。',
        'targetGloss': '我/写字/作业/帮助（我）/需要2/。',
        'passScore': 62,
        'tips': '重点完成“我、作业、帮助、需要”的顺序。',
    },
    {
        'id': '3',
        'title': '物品价格',
        'targetChinese': '这东西贵。',
        'targetGloss': '这/贵/。',
        'passScore': 65,
        'tips': '短句关卡，要求关键 gloss 命中。',
    },
    {
        'id': '4',
        'title': '车辆加油',
        'targetChinese': '可以帮我的车加满油吗？',
        'targetGloss': '帮助（我）/车/加/油/满/可以/？',
        'passScore': 60,
        'tips': '长句会更难，评分会同时看中文和 gloss 重合。',
    },
    {
        'id': '5',
        'title': '询问位置',
        'targetChinese': '请问卧室在哪里？',
        'targetGloss': '卧室/在/哪/？',
        'passScore': 60,
        'tips': '参考便携包内演示样本，可用于课堂展示。',
    },
    {
        'id': '6',
        'title': '住房场景',
        'targetChinese': '房子还没弄好。',
        'targetGloss': '房/建设/好/没/。',
        'passScore': 60,
        'tips': '重点命中“房、好、没”。',
    },
]
SINGLE_WORD_CHALLENGE_WORDS = [
    '我', '你', '他', '她', '家', '爱', '好', '来', '去', '看',
    '听', '说', '学', '问', '答', '吃', '喝', '水', '饭', '茶',
    '书', '笔', '门', '车', '路', '钱', '买', '卖', '大', '小',
    '多', '少', '快', '慢', '高', '低', '新', '旧', '远', '近',
    '上', '下', '左', '右', '前', '后', '早', '晚', '今', '明',
    '昨', '年', '月', '日', '时', '分', '爸', '妈', '哥', '姐',
    '弟', '妹', '师', '友', '校', '班', '课', '题', '字', '词',
    '手', '眼', '口', '心', '头', '身', '病', '药', '医', '院',
    '冷', '热', '雨', '雪', '风', '云', '红', '蓝', '绿', '黑',
    '白', '黄', '一', '二', '三', '四', '五', '六', '七', '八',
    '九', '十', '男', '女', '开', '关', '坐', '站', '走', '跑',
    '停', '等', '帮', '找', '要', '给', '谢', '请', '懂', '会',
]


def build_single_word_challenge_levels():
    levels = []
    for index, word in enumerate(SINGLE_WORD_CHALLENGE_WORDS, start=1):
        band = '基础单字' if index <= 40 else ('进阶单字' if index <= 80 else '高阶单字')
        pass_score = 55 if index <= 40 else (60 if index <= 80 else 65)
        levels.append({
            'id': str(index),
            'title': f'{band} · {word}',
            'targetChinese': word,
            'targetGloss': word,
            'passScore': pass_score,
            'tips': '单字闯关：保持手部完整入镜，动作结束后停顿半秒，便于系统锁定关键帧。',
        })
    return levels


E2E_CHALLENGE_LEVELS = build_single_word_challenge_levels()
E2E_CHALLENGE_BY_ID = {level['id']: level for level in E2E_CHALLENGE_LEVELS}

SHOUYU_ROOT = Path(os.environ.get("SHOUYU_ROOT", PACKAGE_ROOT / "ml" / "shouyu_project")).resolve()
E2E_CHECKPOINT_PATH = SHOUYU_ROOT / "output" / "best_e2e_by_wer.pt"
S2G_CHECKPOINT_PATH = SHOUYU_ROOT / "output" / "best_s2g.pt"
G2T_CHECKPOINT_PATH = SHOUYU_ROOT / "output" / "best_g2t.pt"
KEYWORD_CHECKPOINT_PATH = SHOUYU_ROOT / "output" / "best_keyword_classifier.pt"
HOLISTIC_LANDMARKER_PATH = SHOUYU_ROOT / "models" / "holistic_landmarker.task"
E2E_MODEL_CACHE = {
    'model': None,
    'torch': None,
    'device': None,
    'idx_to_token': None,
    'blank_idx': None,
    'ignored_token_ids': None,
    'dataset_module': None,
    'error': '',
}
REALTIME_KEYWORD_CACHE = {
    'recognizer': None,
    'landmarker': None,
    'compose_sentence': None,
    'extract_one_frame': None,
    'error': '',
    'frames': 0,
}
ADAPTER_ROOT = SHOUYU_ROOT / "tools" / "corrnet_adapter"
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

try:
    from pipeline import translate_video
except Exception as exc:
    translate_video = None
    PIPELINE_IMPORT_ERROR = str(exc)
else:
    PIPELINE_IMPORT_ERROR = ""

# ========================================================
# 工具函数
# ========================================================

def run_async_tts(text, voice, output_path):
    """在子线程运行异步 TTS"""
    import edge_tts
    asyncio.run(edge_tts.Communicate(text, voice).save(output_path))


def public_demo_sample(sample):
    return {
        'id': sample['id'],
        'title': sample['title'],
        'split': sample['split'],
        'translator': sample['translator'],
        'videoName': os.path.basename(sample['videoPath']),
        'referenceChinese': sample['referenceChinese'],
        'referenceGloss': sample['referenceGloss'],
        'note': sample['note'],
        'available': os.path.exists(sample['videoPath']),
    }


def valid_dataset_split(split):
    return split if split in {'dev', 'train', 'test'} else 'dev'


def dataset_video_path(split, translator, number):
    split = valid_dataset_split(split)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", translator or ""):
        return None
    if not re.fullmatch(r"(dev|train|test)-\d{5}", number or ""):
        return None
    path = (CE_CSL_ROOT / "video" / split / translator / f"{number}.mp4").resolve()
    try:
        path.relative_to((CE_CSL_ROOT / "video").resolve())
    except ValueError:
        return None
    return path


def public_dataset_sample(row, split):
    number = (row.get("Number") or "").strip()
    translator = (row.get("Translator") or "").strip()
    video_path = dataset_video_path(split, translator, number)
    return {
        'id': number,
        'number': number,
        'split': split,
        'translator': translator,
        'chinese': (row.get("Chinese Sentences") or "").strip(),
        'gloss': (row.get("Gloss") or "").strip(),
        'note': (row.get("Note") or "").strip(),
        'available': bool(video_path and video_path.exists()),
        'videoUrl': f"/api/learning/dataset-video/{split}/{translator}/{number}" if video_path and video_path.exists() else "",
    }


def typical_dataset_sample_score(item):
    chinese = item.get('chinese', '')
    gloss = item.get('gloss', '')
    note = item.get('note', '')
    gloss_tokens = split_gloss(gloss)
    score = 0
    if chinese:
        score += 30
    if gloss_tokens:
        score += 30
    if note:
        score += 5

    chinese_len = len(chinese)
    gloss_count = len(gloss_tokens)
    score += max(0, 20 - abs(chinese_len - 12))
    score += max(0, 15 - abs(gloss_count - 5) * 3)
    if 6 <= chinese_len <= 24:
        score += 10
    if 2 <= gloss_count <= 8:
        score += 10
    return score


def read_dataset_samples(split='dev', limit=25, query='', offset=0):
    split = valid_dataset_split(split)
    limit = max(1, min(int(limit or 25), 80))
    offset = max(0, int(offset or 0))
    query = (query or "").strip().lower()
    label_path = CE_CSL_ROOT / "label" / f"{split}.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"CE-CSL label file not found: {label_path}")
    candidates = []
    with open(label_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            haystack = " ".join([
                row.get("Number", ""),
                row.get("Translator", ""),
                row.get("Chinese Sentences", ""),
                row.get("Gloss", ""),
                row.get("Note", ""),
            ]).lower()
            if query and query not in haystack:
                continue
            item = public_dataset_sample(row, split)
            if item['available']:
                candidates.append(item)
    ranked = sorted(
        candidates,
        key=lambda item: (
            -typical_dataset_sample_score(item),
            item.get('translator', ''),
            item.get('number', ''),
        ),
    )
    items = ranked[offset:offset + limit]
    return {
        'items': items,
        'split': split,
        'limit': limit,
        'offset': offset,
        'nextOffset': offset + len(items),
        'query': query,
        'matched': len(ranked),
        'datasetRoot': str(CE_CSL_ROOT),
        'available': CE_CSL_ROOT.exists(),
    }


def create_recognition_task(task_id, pipeline_result, user_id, device_id, video_path, extra=None):
    now = datetime.now().isoformat()
    task = {
        'taskId': task_id,
        'status': pipeline_result.get('status', 'failed'),
        'userId': user_id,
        'deviceId': device_id,
        'videoPath': video_path,
        'gloss': pipeline_result.get('gloss', ''),
        'result': pipeline_result.get('result', ''),
        'confidence': pipeline_result.get('confidence', 0.0),
        'latencyMs': pipeline_result.get('latencyMs', 0),
        'model': pipeline_result.get('model', 'unknown'),
        'fallback': pipeline_result.get('fallback', False),
        'rawText': pipeline_result.get('rawText', ''),
        'fallbackReason': pipeline_result.get('fallbackReason', ''),
        'error': pipeline_result.get('error', ''),
        'createTime': now,
        'completeTime': now
    }
    if extra:
        task.update(extra)
    tasks[task_id] = task
    conversation_history.append({
        'taskId': task_id,
        'role': 'user',
        'gloss': task['gloss'],
        'content': task['result'],
        'time': now,
        'deviceId': device_id,
    })
    record_user_task(user_id, task)
    return task


def normalize_chinese(text):
    return re.sub(r"[\s，。！？,.!?；;：:、（）()《》“”\"']", "", text or "")


def split_gloss(gloss):
    return [
        token.strip()
        for token in re.split(r"[/\s]+", gloss or "")
        if token.strip() and token.strip() not in {"<pad>", "<bos>", "<eos>", "<blank>"}
    ]


def sequence_overlap_score(pred_tokens, target_tokens):
    if not target_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    target_set = set(target_tokens)
    if not pred_set:
        return 0.0
    precision = len(pred_set & target_set) / max(len(pred_set), 1)
    recall = len(pred_set & target_set) / max(len(target_set), 1)
    if precision + recall == 0:
        return 0.0
    return 100.0 * (2 * precision * recall) / (precision + recall)


def char_overlap_score(pred_text, target_text):
    pred = normalize_chinese(pred_text)
    target = normalize_chinese(target_text)
    if not target or not pred:
        return 0.0
    pred_chars = set(pred)
    target_chars = set(target)
    precision = len(pred_chars & target_chars) / max(len(pred_chars), 1)
    recall = len(pred_chars & target_chars) / max(len(target_chars), 1)
    if precision + recall == 0:
        return 0.0
    return 100.0 * (2 * precision * recall) / (precision + recall)


def load_community_data():
    if COMMUNITY_DATA_FILE.exists():
        try:
            with open(COMMUNITY_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('posts', [])
        except Exception:
            pass
    return []


def save_community_data(posts):
    with open(COMMUNITY_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump({'posts': posts}, f, ensure_ascii=False, indent=2)


def load_json_file(path, default):
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json_file(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_username(username):
    return re.sub(r"\s+", "", (username or "").strip())


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', (password or '').encode('utf-8'), salt.encode('utf-8'), 200000)
    return salt, digest.hex()


def verify_password(password, salt, stored_hash):
    _, digest = password_hash(password, salt)
    return hmac.compare_digest(digest, stored_hash or '')


def public_user(user):
    if not user:
        return None
    return {
        'username': user.get('username'),
        'createdAt': user.get('createdAt'),
        'lastLoginAt': user.get('lastLoginAt'),
    }


def load_users():
    data = load_json_file(USERS_DATA_FILE, {'users': []})
    return data.get('users', [])


def save_users(users):
    save_json_file(USERS_DATA_FILE, {'users': users})


def find_user(username):
    username = normalize_username(username)
    for user in load_users():
        if user.get('username') == username:
            return user
    return None


def load_sessions():
    data = load_json_file(SESSIONS_DATA_FILE, {'sessions': {}})
    return data.get('sessions', {})


def save_sessions(sessions):
    save_json_file(SESSIONS_DATA_FILE, {'sessions': sessions})


def auth_token_from_request():
    header = request.headers.get('Authorization', '')
    if header.lower().startswith('bearer '):
        return header.split(' ', 1)[1].strip()
    return request.headers.get('X-Huanyu-Token', '').strip() or request.args.get('token', '').strip()


def current_user():
    token = auth_token_from_request()
    if not token:
        return None
    session = load_sessions().get(token)
    if not session:
        return None
    return find_user(session.get('username'))


def require_user():
    user = current_user()
    if not user:
        return None, (jsonify({'code': 401, 'message': '请先登录后再使用该功能'}), 401)
    return user, None


def safe_user_filename(username):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", username or "unknown")


def user_usage_path(username):
    return USER_USAGE_DIR / f"{safe_user_filename(username)}.json"


def load_user_usage(username):
    return load_json_file(user_usage_path(username), {'conversation': [], 'tasks': []})


def save_user_usage(username, usage):
    save_json_file(user_usage_path(username), usage)


def record_user_task(username, task):
    if not username:
        return
    usage = load_user_usage(username)
    usage.setdefault('tasks', []).append(task)
    usage.setdefault('conversation', []).append({
        'taskId': task.get('taskId'),
        'role': 'user',
        'gloss': task.get('gloss'),
        'content': task.get('result'),
        'time': task.get('completeTime') or task.get('createTime'),
        'deviceId': task.get('deviceId'),
    })
    usage['tasks'] = usage['tasks'][-200:]
    usage['conversation'] = usage['conversation'][-200:]
    save_user_usage(username, usage)


HAND_KEYPOINT_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17],
]


def build_synthetic_hand_points(center_x, center_y, scale, phase=0.0):
    wrist = (center_x, center_y + scale * 0.34)
    fingers = [
        (-0.28, -0.04, 0.78),
        (-0.14, -0.23, 1.0),
        (0.00, -0.28, 1.06),
        (0.15, -0.23, 0.98),
        (0.29, -0.08, 0.82),
    ]
    points = [{'x': round(wrist[0], 4), 'y': round(wrist[1], 4), 'score': 0.88}]
    for finger_index, (root_x, root_y, length) in enumerate(fingers):
        for joint in range(4):
            t = (joint + 1) / 4
            bend = math.sin(phase + finger_index * 0.8 + joint * 0.35) * 0.025
            x = center_x + scale * (root_x * t + bend)
            y = center_y + scale * (0.30 + root_y * t - length * 0.56 * t)
            points.append({'x': round(max(0.02, min(0.98, x)), 4), 'y': round(max(0.02, min(0.98, y)), 4), 'score': 0.82})
    return points


def synthetic_keypoint_preview(seed_text=''):
    digest = hashlib.sha1(str(seed_text).encode('utf-8', errors='ignore')).hexdigest()
    phase = int(digest[:4], 16) / 65535 * math.pi
    return {
        'source': 'gesture-estimate',
        'connections': HAND_KEYPOINT_CONNECTIONS,
        'hands': [
            {'label': 'left', 'points': build_synthetic_hand_points(0.35, 0.55, 0.52, phase)},
            {'label': 'right', 'points': build_synthetic_hand_points(0.66, 0.54, 0.50, phase + 0.9)},
        ],
    }


def keypoint_preview_from_vector(keypoints):
    if keypoints is None:
        return None
    try:
        values = keypoints.detach().cpu().numpy().reshape(-1).tolist()
    except Exception:
        try:
            values = list(keypoints.reshape(-1))
        except Exception:
            values = list(keypoints)
    if len(values) < 258:
        return None

    def read_hand(label, offset):
        points = []
        visible = 0
        for index in range(21):
            base = offset + index * 3
            x = float(values[base] or 0)
            y = float(values[base + 1] or 0)
            score = 0.92 if x or y else 0.0
            if score:
                visible += 1
            points.append({'x': round(x, 4), 'y': round(y, 4), 'score': score})
        return {'label': label, 'points': points, 'visible': visible}

    hands = [read_hand('left', 33 * 4), read_hand('right', 33 * 4 + 21 * 3)]
    if not any(hand['visible'] >= 3 for hand in hands):
        return None
    return {'source': 'model-keypoints', 'connections': HAND_KEYPOINT_CONNECTIONS, 'hands': hands}


def extract_challenge_keypoint_preview(video_path, level=None):
    try:
        cache = ensure_realtime_keyword_loaded()
        import cv2
        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 3:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            raise ValueError('no readable frame')
        keypoints = cache['extract_one_frame'](cache['landmarker'], frame)
        preview = keypoint_preview_from_vector(keypoints)
        if preview:
            return preview
    except Exception:
        pass
    return synthetic_keypoint_preview((level or {}).get('targetChinese', ''))


def score_challenge(level, pipeline_result):
    pred_gloss = pipeline_result.get('gloss', '')
    pred_text = pipeline_result.get('result', '')
    gloss_score = sequence_overlap_score(split_gloss(pred_gloss), split_gloss(level['targetGloss']))
    text_score = char_overlap_score(pred_text, level['targetChinese'])
    score = round(max(gloss_score, text_score * 0.8), 1)
    passed = score >= level['passScore']
    if passed:
        feedback = '动作关键序列已命中，可以进入下一关。'
    elif score >= 45:
        feedback = '接近目标，建议放慢动作并保持手部在画面中心。'
    else:
        feedback = '识别结果与目标差距较大，建议重录一段光线充足、身体完整入镜的视频。'
    return {
        'score': score,
        'passed': passed,
        'feedback': feedback,
        'glossScore': round(gloss_score, 1),
        'textScore': round(text_score, 1),
    }


def public_challenge_level(level):
    return {
        'id': level['id'],
        'title': level['title'],
        'targetChinese': level['targetChinese'],
        'targetGloss': level['targetGloss'],
        'passScore': level['passScore'],
        'tips': level['tips'],
    }


def e2e_model_status():
    return {
        'directE2E': {
            'checkpoint': str(E2E_CHECKPOINT_PATH),
            'available': E2E_CHECKPOINT_PATH.exists(),
            'name': 'best_e2e_by_wer.pt',
        },
        'twoStage': {
            's2gCheckpoint': str(S2G_CHECKPOINT_PATH),
            'g2tCheckpoint': str(G2T_CHECKPOINT_PATH),
            's2gAvailable': S2G_CHECKPOINT_PATH.exists(),
            'g2tAvailable': G2T_CHECKPOINT_PATH.exists(),
        },
        'bestWer': '69.42%',
        'notes': '直接 E2E 模型用于闯关 gloss 评测；若运行环境缺少 torch/cv2，会返回明确 fallback。',
        'lastError': E2E_MODEL_CACHE.get('error', ''),
    }


def realtime_keyword_status():
    available = KEYWORD_CHECKPOINT_PATH.exists() and HOLISTIC_LANDMARKER_PATH.exists()
    return {
        'available': bool(available),
        'checkpoint': str(KEYWORD_CHECKPOINT_PATH),
        'checkpointAvailable': KEYWORD_CHECKPOINT_PATH.exists(),
        'landmarker': str(HOLISTIC_LANDMARKER_PATH),
        'landmarkerAvailable': HOLISTIC_LANDMARKER_PATH.exists(),
        'model': 'best_keyword_classifier.pt',
        'mode': 'keyword-realtime',
        'framesProcessed': REALTIME_KEYWORD_CACHE.get('frames', 0),
        'error': REALTIME_KEYWORD_CACHE.get('error', ''),
    }


def ensure_realtime_keyword_loaded():
    if REALTIME_KEYWORD_CACHE['recognizer'] is not None and REALTIME_KEYWORD_CACHE['landmarker'] is not None:
        return REALTIME_KEYWORD_CACHE
    if not KEYWORD_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f'Keyword checkpoint not found: {KEYWORD_CHECKPOINT_PATH}')
    if not HOLISTIC_LANDMARKER_PATH.exists():
        raise FileNotFoundError(f'Holistic landmarker not found: {HOLISTIC_LANDMARKER_PATH}')
    if str(SHOUYU_ROOT) not in sys.path:
        sys.path.insert(0, str(SHOUYU_ROOT))

    import cv2
    import numpy as np
    import torch
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options as mp_base_options
    from realtime_inference import extract_one_frame
    from realtime_keyword_inference import StreamingKeywordRecognizer, compose_demo_sentence

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recognizer = StreamingKeywordRecognizer.from_checkpoint(
        KEYWORD_CHECKPOINT_PATH,
        device=device,
        stabilize=True,
        top_k=5,
    )
    landmarker_bytes = HOLISTIC_LANDMARKER_PATH.read_bytes()
    options = vision.HolisticLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_buffer=landmarker_bytes),
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    REALTIME_KEYWORD_CACHE.update({
        'recognizer': recognizer,
        'landmarker': vision.HolisticLandmarker.create_from_options(options),
        'compose_sentence': compose_demo_sentence,
        'extract_one_frame': extract_one_frame,
        'cv2': cv2,
        'np': np,
        'error': '',
    })
    return REALTIME_KEYWORD_CACHE


def decode_realtime_frame(frame_data):
    if not frame_data:
        return None
    payload = frame_data.split(',', 1)[1] if ',' in frame_data else frame_data
    cache = ensure_realtime_keyword_loaded()
    raw = base64.b64decode(payload)
    arr = cache['np'].frombuffer(raw, dtype=cache['np'].uint8)
    return cache['cv2'].imdecode(arr, cache['cv2'].IMREAD_COLOR)


def fallback_realtime_result(reason):
    return {
        'status': 'fallback',
        'keywords': [],
        'sentence': '实时关键词模型暂不可用，可继续使用上传视频识别。',
        'confidence': 0.0,
        'latencyMs': 0,
        'model': 'keyword realtime fallback',
        'fallback': True,
        'fallbackReason': reason,
        'gloss': '',
        'keypointPreview': synthetic_keypoint_preview(reason),
    }


def run_realtime_keyword_frame(frame_data=None, keypoints=None):
    started = time.perf_counter()
    cache = ensure_realtime_keyword_loaded()
    recognizer = cache['recognizer']
    if keypoints is not None:
        kp = cache['np'].asarray(keypoints, dtype=cache['np'].float32)
    else:
        frame = decode_realtime_frame(frame_data)
        if frame is None:
            raise ValueError('missing frameData')
        kp = cache['extract_one_frame'](cache['landmarker'], frame)
        if kp is None:
            raise ValueError('no landmarks detected')
    preds = recognizer.add_frame(kp)
    REALTIME_KEYWORD_CACHE['frames'] = REALTIME_KEYWORD_CACHE.get('frames', 0) + 1
    keywords = []
    if preds:
        keywords = [
            {'token': pred.token, 'confidence': round(float(pred.confidence), 4)}
            for pred in preds
        ]
    tokens = [item['token'] for item in keywords]
    sentence = cache['compose_sentence'](tokens) if tokens else '正在观察动作...'
    return {
        'status': 'ok',
        'keywords': keywords,
        'sentence': sentence,
        'confidence': keywords[0]['confidence'] if keywords else 0.0,
        'latencyMs': int((time.perf_counter() - started) * 1000),
        'model': 'keyword realtime best_keyword_classifier.pt',
        'fallback': False,
        'fallbackReason': '',
        'gloss': '/'.join(tokens),
        'keypointPreview': keypoint_preview_from_vector(kp) or synthetic_keypoint_preview('/'.join(tokens)),
    }


def ensure_e2e_model_loaded():
    if E2E_MODEL_CACHE['model'] is not None:
        return E2E_MODEL_CACHE
    if not E2E_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f'E2E checkpoint not found: {E2E_CHECKPOINT_PATH}')
    if str(SHOUYU_ROOT) not in sys.path:
        sys.path.insert(0, str(SHOUYU_ROOT))

    import importlib
    import torch
    from model_e2e import E2EModel

    dataset_module = importlib.import_module('dataset_e2e')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(E2E_CHECKPOINT_PATH, map_location=device)
    vocab = checkpoint.get('vocab') or {}
    idx_to_token = checkpoint.get('idx_to_token') or {v: k for k, v in vocab.items()}
    blank_idx = vocab.get('<blank>', 4)
    model = E2EModel(
        vocab_size=len(vocab),
        blank_idx=blank_idx,
        pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    E2E_MODEL_CACHE.update({
        'model': model,
        'torch': torch,
        'device': device,
        'idx_to_token': idx_to_token,
        'blank_idx': blank_idx,
        'ignored_token_ids': {vocab[tok] for tok in ('<pad>', '<bos>', '<eos>') if tok in vocab},
        'dataset_module': dataset_module,
        'error': '',
    })
    return E2E_MODEL_CACHE


def run_direct_e2e_video(video_path):
    cache = ensure_e2e_model_loaded()
    torch = cache['torch']
    dataset_module = cache['dataset_module']
    frames = dataset_module.E2EVideoDataset._load_frames(video_path, dataset_module.NUM_FRAMES)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
    height, width = frames.shape[2], frames.shape[3]
    top = max((height - dataset_module.CROP_SIZE) // 2, 0)
    left = max((width - dataset_module.CROP_SIZE) // 2, 0)
    frames = frames[:, :, top:top + dataset_module.CROP_SIZE, left:left + dataset_module.CROP_SIZE]
    frames = (frames - dataset_module.IMAGENET_MEAN) / dataset_module.IMAGENET_STD
    frames = frames.unsqueeze(0).to(cache['device'])
    valid_mask = torch.ones((1, frames.shape[1]), dtype=torch.bool, device=cache['device'])
    input_lengths = torch.tensor([frames.shape[1]], dtype=torch.long, device=cache['device'])
    started = time.perf_counter()
    with torch.no_grad():
        log_probs = cache['model'].forward_final(frames, valid_mask).float()
        decoded = cache['model'].decode(
            log_probs,
            input_lengths,
            cache['idx_to_token'],
            cache['blank_idx'],
            ignored_token_ids=cache['ignored_token_ids'],
        )[0]
    gloss = '/'.join(decoded)
    return {
        'status': 'completed',
        'gloss': gloss,
        'result': 'E2E gloss: ' + gloss if gloss else 'E2E 未解码到有效 gloss',
        'confidence': 0.0,
        'latencyMs': int((time.perf_counter() - started) * 1000),
        'model': 'direct-e2e-ctc best_e2e_by_wer.pt',
        'fallback': False,
        'rawText': gloss,
        'fallbackReason': '',
        'error': '',
    }


def run_e2e_challenge_inference(video_path, level):
    try:
        return run_direct_e2e_video(video_path)
    except Exception as exc:
        E2E_MODEL_CACHE['error'] = str(exc)
        if translate_video is not None:
            pipeline_result = translate_video(video_path)
            if pipeline_result.get('status') != 'failed' or pipeline_result.get('result'):
                return {
                    **pipeline_result,
                    'model': f"fallback-corrnet-g2t after e2e_error: {pipeline_result.get('model', 'unknown')}",
                    'fallback': True,
                    'fallbackReason': f"direct_e2e_failed: {exc}",
                }
        return {
            'status': 'demo',
            'gloss': level['targetGloss'],
            'result': level['targetChinese'],
            'confidence': 0.0,
            'latencyMs': 0,
            'model': 'challenge demo fallback',
            'fallback': True,
            'rawText': '',
            'fallbackReason': f"direct_e2e_failed: {exc}",
            'error': str(exc),
        }

# ========================================================
# 0. E2E 闯关学习模块
# ========================================================

# ========================================================
# Auth
# ========================================================

@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    data = request.get_json(silent=True) or {}
    username = normalize_username(data.get('username'))
    password = data.get('password') or ''
    if not username:
        return jsonify({'code': 400, 'message': '请输入用户名'}), 400
    if not password:
        return jsonify({'code': 400, 'message': '请设置密码'}), 400
    users = load_users()
    if any(user.get('username') == username for user in users):
        return jsonify({'code': 409, 'message': '用户名已存在，请直接登录'}), 409
    salt, digest = password_hash(password)
    now = datetime.now().isoformat()
    user = {
        'username': username,
        'passwordSalt': salt,
        'passwordHash': digest,
        'createdAt': now,
        'lastLoginAt': now,
    }
    users.append(user)
    save_users(users)
    token = secrets.token_urlsafe(32)
    sessions = load_sessions()
    sessions[token] = {'username': username, 'createdAt': now}
    save_sessions(sessions)
    save_user_usage(username, load_user_usage(username))
    return jsonify({'code': 200, 'message': '注册成功', 'data': {'token': token, 'user': public_user(user)}})


@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    data = request.get_json(silent=True) or {}
    username = normalize_username(data.get('username'))
    password = data.get('password') or ''
    user = find_user(username)
    if not user or not verify_password(password, user.get('passwordSalt'), user.get('passwordHash')):
        return jsonify({'code': 401, 'message': '用户名或密码错误'}), 401
    users = load_users()
    now = datetime.now().isoformat()
    for item in users:
        if item.get('username') == username:
            item['lastLoginAt'] = now
            user = item
            break
    save_users(users)
    token = secrets.token_urlsafe(32)
    sessions = load_sessions()
    sessions[token] = {'username': username, 'createdAt': now}
    save_sessions(sessions)
    return jsonify({'code': 200, 'message': '登录成功', 'data': {'token': token, 'user': public_user(user)}})


@app.route('/api/auth/me', methods=['GET'])
def auth_me():
    user = current_user()
    return jsonify({'code': 200, 'message': '查询成功', 'data': {'user': public_user(user), 'guest': user is None}})


@app.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    token = auth_token_from_request()
    sessions = load_sessions()
    if token and token in sessions:
        sessions.pop(token, None)
        save_sessions(sessions)
    return jsonify({'code': 200, 'message': '已退出登录'})


@app.route('/api/e2e/challenge/status', methods=['GET'])
def get_e2e_challenge_status():
    user, error = require_user()
    if error:
        return error
    return jsonify({
        'code': 200,
        'message': '获取成功',
        'data': e2e_model_status(),
    })


@app.route('/api/recognition/realtime/status', methods=['GET'])
def get_realtime_recognition_status():
    user, error = require_user()
    if error:
        return error
    data = realtime_keyword_status()
    return jsonify({
        'code': 200,
        'message': '获取成功',
        'data': data,
    })


@app.route('/api/recognition/realtime/frame', methods=['POST'])
def submit_realtime_frame():
    user, error = require_user()
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    started = time.perf_counter()
    try:
        result = run_realtime_keyword_frame(
            frame_data=payload.get('frameData'),
            keypoints=payload.get('keypoints'),
        )
    except Exception as exc:
        REALTIME_KEYWORD_CACHE['error'] = str(exc)
        result = fallback_realtime_result(str(exc))
        result['latencyMs'] = int((time.perf_counter() - started) * 1000)
    return jsonify({
        'code': 200,
        'message': '实时识别完成',
        'data': result,
    })


@app.route('/api/e2e/challenge/levels', methods=['GET'])
def list_e2e_challenge_levels():
    user, error = require_user()
    if error:
        return error
    return jsonify({
        'code': 200,
        'message': '获取成功',
        'data': {
            'items': [public_challenge_level(level) for level in E2E_CHALLENGE_LEVELS]
        },
    })


@app.route('/api/learning/dataset-samples', methods=['GET'])
def list_learning_dataset_samples():
    user, error = require_user()
    if error:
        return error
    try:
        data = read_dataset_samples(
            split=request.args.get('split', 'dev'),
            limit=request.args.get('limit', 25),
            query=request.args.get('query', ''),
            offset=request.args.get('offset', 0),
        )
    except Exception as exc:
        return jsonify({'code': 500, 'message': str(exc), 'data': {'items': []}}), 500
    return jsonify({
        'code': 200,
        'message': '获取成功',
        'data': data,
    })


@app.route('/api/learning/dataset-video/<split>/<translator>/<number>', methods=['GET'])
def get_learning_dataset_video(split, translator, number):
    user, error = require_user()
    if error:
        return error
    video_path = dataset_video_path(split, translator, number)
    if video_path is None or not video_path.exists():
        return jsonify({'code': 404, 'message': '训练集视频不存在'}), 404
    return send_file(str(video_path), mimetype='video/mp4', conditional=True)


@app.route('/api/e2e/challenge/submit', methods=['POST'])
def submit_e2e_challenge():
    user, error = require_user()
    if error:
        return error
    if 'video' not in request.files:
        return jsonify({'code': 400, 'message': '没有视频文件'}), 400
    level_id = request.form.get('levelId', '1')
    level = E2E_CHALLENGE_BY_ID.get(level_id)
    if level is None:
        return jsonify({'code': 404, 'message': '关卡不存在'}), 404

    video = request.files['video']
    task_id = str(uuid.uuid4())
    filename = f"{task_id}_{video.filename}" if video.filename else f"{task_id}.webm"
    video_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
    video.save(video_path)

    pipeline_result = run_e2e_challenge_inference(video_path, level)
    keypoint_preview = extract_challenge_keypoint_preview(video_path, level)
    scoring = score_challenge(level, pipeline_result)
    task = create_recognition_task(
        task_id,
        pipeline_result,
        user['username'],
        request.form.get('deviceId', 'challenge-browser'),
        video_path,
        extra={
            'challengeLevelId': level['id'],
            'referenceChinese': level['targetChinese'],
            'referenceGloss': level['targetGloss'],
            'score': scoring['score'],
            'passed': scoring['passed'],
            'feedback': scoring['feedback'],
        },
    )

    return jsonify({
        'code': 200,
        'message': '闯关评测完成',
        'data': {
            'taskId': task['taskId'],
            'levelId': level['id'],
            'title': level['title'],
            'targetChinese': level['targetChinese'],
            'targetGloss': level['targetGloss'],
            'status': task['status'],
            'gloss': task['gloss'],
            'result': task['result'],
            'confidence': task['confidence'],
            'latencyMs': task['latencyMs'],
            'model': task['model'],
            'fallback': task['fallback'],
            'fallbackReason': task['fallbackReason'],
            'error': task['error'],
            'score': scoring['score'],
            'passed': scoring['passed'],
            'feedback': scoring['feedback'],
            'glossScore': scoring['glossScore'],
            'textScore': scoring['textScore'],
            'keypointPreview': keypoint_preview,
        },
    })


# ========================================================
# 1. 手语识别模块
# ========================================================

@app.route('/api/recognition/upload', methods=['POST'])
def upload_video():
    """
    接收视频文件，创建识别任务
    
    Form Data:
        - video: 视频文件
        - userId: 用户ID
        - deviceId: 设备ID
    
    返回:
        - taskId: 任务ID
        - status: pending/processing/completed/failed
    """
    user, error = require_user()
    if error:
        return error
    if 'video' not in request.files:
        return jsonify({'code': 400, 'message': '没有视频文件'}), 400
    
    video = request.files['video']
    user_id = user['username']
    device_id = request.form.get('deviceId', 'unknown')
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 保存视频
    filename = f"{task_id}_{video.filename}" if video.filename else f"{task_id}.mp4"
    video_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
    video.save(video_path)
    
    started = time.perf_counter()
    if translate_video is None:
        pipeline_result = {
            'status': 'demo',
            'gloss': '我/需要/帮助/。',
            'result': '我需要帮助。',
            'confidence': 0.0,
            'latencyMs': int((time.perf_counter() - started) * 1000),
            'model': 'demo fallback',
            'fallback': True,
            'rawText': '',
            'fallbackReason': 'pipeline_import_error',
            'error': PIPELINE_IMPORT_ERROR,
        }
    else:
        pipeline_result = translate_video(video_path)
        if pipeline_result.get('status') == 'failed' and not pipeline_result.get('result'):
            pipeline_result = {
                **pipeline_result,
                'status': 'demo',
                'gloss': '我/需要/帮助/。',
                'result': '我需要帮助。',
                'confidence': 0.0,
                'model': 'demo fallback',
                'fallback': True,
                'rawText': pipeline_result.get('rawText', ''),
                'fallbackReason': pipeline_result.get('fallbackReason', 'pipeline_failed'),
            }

    now = datetime.now().isoformat()

    # 创建任务记录
    tasks[task_id] = {
        'taskId': task_id,
        'status': pipeline_result.get('status', 'failed'),
        'userId': user_id,
        'deviceId': device_id,
        'videoPath': video_path,
        'gloss': pipeline_result.get('gloss', ''),
        'result': pipeline_result.get('result', ''),
        'confidence': pipeline_result.get('confidence', 0.0),
        'latencyMs': pipeline_result.get('latencyMs', 0),
        'model': pipeline_result.get('model', 'unknown'),
        'fallback': pipeline_result.get('fallback', False),
        'rawText': pipeline_result.get('rawText', ''),
        'fallbackReason': pipeline_result.get('fallbackReason', ''),
        'error': pipeline_result.get('error', ''),
        'createTime': now,
        'completeTime': now
    }
    conversation_history.append({
        'taskId': task_id,
        'role': 'user',
        'gloss': tasks[task_id]['gloss'],
        'content': tasks[task_id]['result'],
        'time': now,
        'deviceId': device_id,
    })
    
    return jsonify({
        'code': 200,
        'message': '识别完成' if tasks[task_id]['status'] == 'completed' else '演示或降级结果',
        'data': {
            'taskId': task_id,
            'status': tasks[task_id]['status'],
            'gloss': tasks[task_id]['gloss'],
            'result': tasks[task_id]['result'],
            'confidence': tasks[task_id]['confidence'],
            'latencyMs': tasks[task_id]['latencyMs'],
            'model': tasks[task_id]['model'],
            'fallback': tasks[task_id]['fallback'],
            'rawText': tasks[task_id]['rawText'],
            'fallbackReason': tasks[task_id]['fallbackReason'],
            'error': tasks[task_id]['error'],
        }
    })


@app.route('/api/recognition/demo-samples', methods=['GET'])
def list_demo_samples():
    user, error = require_user()
    if error:
        return error
    return jsonify({
        'code': 200,
        'message': '获取成功',
        'data': {
            'items': [public_demo_sample(sample) for sample in DEMO_SAMPLES]
        }
    })


@app.route('/api/recognition/demo-samples/<sample_id>/run', methods=['POST'])
def run_demo_sample(sample_id):
    user, error = require_user()
    if error:
        return error
    sample = DEMO_SAMPLE_BY_ID.get(sample_id)
    if sample is None:
        return jsonify({'code': 404, 'message': '演示样本不存在'}), 404
    if not os.path.exists(sample['videoPath']):
        return jsonify({'code': 404, 'message': '演示视频不存在'}), 404

    started = time.perf_counter()
    if translate_video is None:
        pipeline_result = {
            'status': 'demo',
            'gloss': sample['referenceGloss'],
            'result': sample['referenceChinese'],
            'confidence': 0.0,
            'latencyMs': int((time.perf_counter() - started) * 1000),
            'model': 'demo fallback',
            'fallback': True,
            'rawText': '',
            'fallbackReason': 'pipeline_import_error',
            'error': PIPELINE_IMPORT_ERROR,
        }
    else:
        pipeline_result = translate_video(sample['videoPath'])
        if pipeline_result.get('status') == 'failed' and not pipeline_result.get('result'):
            pipeline_result = {
                **pipeline_result,
                'status': 'demo',
                'gloss': sample['referenceGloss'],
                'result': sample['referenceChinese'],
                'confidence': 0.0,
                'model': 'demo fallback',
                'fallback': True,
                'rawText': pipeline_result.get('rawText', ''),
                'fallbackReason': pipeline_result.get('fallbackReason', 'pipeline_failed'),
            }

    task_id = str(uuid.uuid4())
    task = create_recognition_task(
        task_id,
        pipeline_result,
        user['username'],
        request.form.get('deviceId', 'curated-demo'),
        sample['videoPath'],
        extra={
            'sampleId': sample['id'],
            'referenceChinese': sample['referenceChinese'],
            'referenceGloss': sample['referenceGloss'],
        },
    )

    return jsonify({
        'code': 200,
        'message': '演示样本运行完成',
        'data': {
            'taskId': task['taskId'],
            'sampleId': sample['id'],
            'title': sample['title'],
            'status': task['status'],
            'gloss': task['gloss'],
            'result': task['result'],
            'confidence': task['confidence'],
            'latencyMs': task['latencyMs'],
            'model': task['model'],
            'fallback': task['fallback'],
            'rawText': task['rawText'],
            'fallbackReason': task['fallbackReason'],
            'referenceChinese': sample['referenceChinese'],
            'referenceGloss': sample['referenceGloss'],
            'error': task['error'],
        }
    })


@app.route('/api/recognition/status/<task_id>', methods=['GET'])
def get_recognition_status(task_id):
    """查询识别任务状态"""
    user, error = require_user()
    if error:
        return error
    if task_id not in tasks:
        return jsonify({'code': 404, 'message': '任务不存在'}), 404
    
    task = tasks[task_id]
    if task.get('userId') != user['username']:
        return jsonify({'code': 403, 'message': '无权查看其他用户任务'}), 403
    return jsonify({
        'code': 200,
        'message': '查询成功',
        'data': {
            'taskId': task['taskId'],
            'status': task['status'],
            'gloss': task.get('gloss'),
            'result': task.get('result'),
            'confidence': task.get('confidence'),
            'latencyMs': task.get('latencyMs'),
            'model': task.get('model'),
            'fallback': task.get('fallback'),
            'rawText': task.get('rawText'),
            'fallbackReason': task.get('fallbackReason'),
            'error': task.get('error'),
            'createTime': task['createTime'],
            'completeTime': task.get('completeTime')
        }
    })


@app.route('/api/recognition/result/<task_id>', methods=['GET'])
def get_recognition_result(task_id):
    """获取识别结果"""
    user, error = require_user()
    if error:
        return error
    if task_id not in tasks:
        return jsonify({'code': 404, 'message': '任务不存在'}), 404
    
    task = tasks[task_id]
    if task.get('userId') != user['username']:
        return jsonify({'code': 403, 'message': '无权查看其他用户任务'}), 403
    if task['status'] != 'completed':
        return jsonify({
            'code': 400,
            'message': f'任务未完成，当前状态: {task["status"]}'
        }), 400
    
    return jsonify({
        'code': 200,
        'message': '获取成功',
        'data': {
            'taskId': task['taskId'],
            'gloss': task.get('gloss'),
            'result': task['result'],
            'confidence': task['confidence'],
            'latencyMs': task.get('latencyMs'),
            'model': task.get('model'),
            'fallback': task.get('fallback'),
            'rawText': task.get('rawText'),
            'fallbackReason': task.get('fallbackReason'),
            'error': task.get('error'),
            'completeTime': task.get('completeTime')
        }
    })


# ========================================================
# 2. 对话分析模块
# ========================================================

@app.route('/api/dialogue/analyze', methods=['POST'])
def analyze_dialogue():
    """
    完整对话分析（情感 + 推荐回复）
    
    请求体:
    {
        "conversationId": "会话ID",
        "messages": [
            {"role": "user", "content": "消息内容", "time": "ISO时间"}
        ]
    }
    """
    data = request.get_json()
    if not data or 'messages' not in data:
        return jsonify({'code': 400, 'message': '缺少消息列表'}), 400
    
    messages = data['messages']
    last_msg = messages[-1]['content'] if messages else ''
    
    # TODO: 调用 TextCNN-BiLSTM + Seq2Seq 模型
    
    # 模拟分析结果
    emotion_scores = {
        'positive': 0.3,
        'neutral': 0.5,
        'negative': 0.2
    }
    
    recommendations = [
        '好的，我明白了',
        '请问还有什么需要帮助的吗？',
        '感谢您的告知',
        '我会尽快处理'
    ]
    
    return jsonify({
        'code': 200,
        'message': '分析完成',
        'data': {
            'emotion': max(emotion_scores, key=emotion_scores.get),
            'emotionScores': emotion_scores,
            'recommendations': recommendations
        }
    })


@app.route('/api/dialogue/emotion', methods=['POST'])
def analyze_emotion():
    """
    单独的情感分析
    
    请求体: {"text": "要分析的文本"}
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'code': 400, 'message': '缺少文本内容'}), 400
    
    text = data['text']
    
    # TODO: 调用 TextCNN-BiLSTM-SelfAttention 模型
    emotion_result = {
        'text': text,
        'emotion': 'neutral',
        'confidence': 0.85,
        'scores': {'positive': 0.2, 'negative': 0.15, 'neutral': 0.65}
    }
    
    return jsonify({
        'code': 200,
        'message': '情感分析完成',
        'data': emotion_result
    })


@app.route('/api/dialogue/recommend', methods=['POST'])
def recommend_reply():
    """
    推荐回复
    
    请求体: {"lastMessage": "最后一条消息", "history": ["历史"]}
    """
    data = request.get_json()
    if not data or 'lastMessage' not in data:
        return jsonify({'code': 400, 'message': '缺少消息'}), 400
    
    # TODO: 调用 Seq2Seq 模型
    recommendations = [
        '好的，我明白了',
        '请问还有什么需要帮助的吗？',
        '感谢您的耐心等待',
        '我会尽快处理这个问题'
    ]
    
    return jsonify({
        'code': 200,
        'message': '推荐生成完成',
        'data': {
            'lastMessage': data['lastMessage'],
            'recommendations': recommendations,
            'count': len(recommendations)
        }
    })


@app.route('/api/conversation/history', methods=['GET'])
def get_conversation_history():
    """获取 MVP 对话记录"""
    user, error = require_user()
    if error:
        return error
    limit = int(request.args.get('limit', 50))
    usage = load_user_usage(user['username'])
    items = usage.get('conversation', [])[-limit:]
    return jsonify({
        'code': 200,
        'message': '查询成功',
        'data': {
            'items': items,
            'count': len(items)
        }
    })


@app.route('/api/device/status', methods=['GET', 'POST'])
def handle_device_status():
    """树莓派设备状态上报与查询"""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        device_status.update({
            'deviceId': data.get('deviceId', device_status['deviceId']),
            'online': bool(data.get('online', True)),
            'camera': data.get('camera', device_status['camera']),
            'battery': data.get('battery', device_status['battery']),
            'lastSeen': datetime.now().isoformat(),
        })
    return jsonify({
        'code': 200,
        'message': '查询成功',
        'data': device_status
    })


@app.route('/api/emergency/alert', methods=['POST'])
def emergency_alert():
    """一键提示或监护者提醒"""
    data = request.get_json(silent=True) or {}
    now = datetime.now().isoformat()
    item = {
        'taskId': f"alert-{uuid.uuid4()}",
        'role': 'system',
        'content': data.get('message', '使用者为听障人士，请您体谅使用者表达不便。'),
        'time': now,
        'deviceId': data.get('deviceId', 'unknown'),
        'alert': True,
    }
    conversation_history.append(item)
    return jsonify({
        'code': 200,
        'message': '提醒已记录',
        'data': item
    })


# ========================================================
# 3. TTS 语音合成模块
# ========================================================

# Edge TTS 支持的中文音色
EDGE_TTS_VOICES = {
    # 女声
    'zh-CN-XiaoxiaoNeural': '晓晓 - 青年女声',
    'zh-CN-XiaoyiNeural': '小艺 - 青年女声',
    'zh-CN-YunxiNeural': '云希 - 青年女声',
    'zh-CN-YunxiaNeural': '云夏 - 青年女声',
    'zh-CN-YunyangNeural': '云扬 - 青年女声（正式）',
    # 男声
    'zh-CN-YunxiNeural-male': '云希 - 青年男声',  # 注意：EdgeTTS 某些版本格式不同
    'zh-CN-YunyangNeural-male': '云扬 - 青年男声',
    # 老年
    'zh-CN-XiaomeiNeural': '小美 - 女声（温和）',
}


@app.route('/api/tts/synthesize', methods=['POST'])
def tts_synthesize():
    """
    文本转语音（使用 Edge TTS）
    
    请求体:
    {
        "text": "要合成的文本",
        "voiceType": "zh-CN-XiaoxiaoNeural",  // 可选，默认晓晓
        "speed": 1.0,                          // 语速 0.5-2.0
        "pitch": 0                             // 音调 -50 到 50
    }
    
    返回: audio/mpeg 音频文件
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'code': 400, 'message': '缺少文本内容'}), 400
    
    text = data['text']
    voice_type = data.get('voiceType', 'zh-CN-XiaoxiaoNeural')
    rate = data.get('speed', 1.0)
    pitch = data.get('pitch', 0)
    
    # 验证音色
    if voice_type not in EDGE_TTS_VOICES:
        return jsonify({
            'code': 400,
            'message': f'不支持的音色: {voice_type}',
            'data': {'availableVoices': list(EDGE_TTS_VOICES.keys())}
        }), 400
    
    # 生成输出路径
    audio_filename = f"tts_{uuid.uuid4()}.mp3"
    audio_path = os.path.join(RESULT_FOLDER, audio_filename)
    
    try:
        # 调用 Edge TTS
        _do_tts_sync(text, voice_type, rate, pitch, audio_path)
        
        return send_file(
            audio_path,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name=audio_filename
        )
    except Exception as e:
        return jsonify({'code': 500, 'message': f'TTS 合成失败: {str(e)}'}), 500


def _do_tts_sync(text, voice, rate, pitch, output_path):
    """同步执行 TTS（启动子线程运行 asyncio）"""
    import edge_tts
    
    # 将 rate 从 0.5-2.0 转换为 EdgeTTS 格式 (+50% 到 +100%)
    edge_rate = f'+{int((rate - 1) * 100)}%'
    edge_pitch = f'+{pitch}Hz' if pitch >= 0 else f'{pitch}Hz'
    
    communicate = edge_tts.Communicate(text, voice, rate=edge_rate, pitch=edge_pitch)
    asyncio.run(communicate.save(output_path))


@app.route('/api/tts/voices', methods=['GET'])
def get_voice_list():
    """获取可用音色列表"""
    return jsonify({
        'code': 200,
        'message': '查询成功',
        'data': {
            'voices': [
                {'id': k, 'name': v}
                for k, v in EDGE_TTS_VOICES.items()
            ]
        }
    })


# ========================================================
# 4. 健康检查
# ========================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """服务健康检查"""
    return jsonify({
        'code': 200,
        'message': '服务正常',
        'data': {
            'service': '唤语算法服务',
            'version': '1.0.0',
            'time': datetime.now().isoformat(),
            'features': {
                'recognition': '手语识别（CorrNet/G2T 或降级演示）',
                'e2eChallenge': 'E2E 闯关评测（best_e2e_by_wer.pt gloss CTC）',
                'dialogue': '对话分析（TODO: 待接入模型）',
                'tts': 'TTS语音合成（已集成Edge TTS）'
            }
        }
    })


# ========================================================
# 5. 社区讨论模块
# ========================================================

CATEGORIES = {
    'bug': '问题反馈',
    'discussion': '讨论交流',
    'showcase': '学习展示',
}


def _public_post(post):
    return {
        'id': post['id'],
        'title': post['title'],
        'content': post['content'],
        'category': post.get('category', 'discussion'),
        'author': post.get('author', {}),
        'createdAt': post['createdAt'],
        'replyCount': len(post.get('replies', [])),
    }


def _public_reply(reply):
    return {
        'id': reply['id'],
        'content': reply['content'],
        'author': reply.get('author', {}),
        'createdAt': reply['createdAt'],
        'replyTo': reply.get('replyTo'),
    }


def _identity_from_request(req_json=None):
    user = current_user()
    if not user:
        return None
    return {'account': user['username'], 'identity': 'user'}


@app.route('/api/community/posts', methods=['GET'])
def community_list_posts():
    category = request.args.get('category', '')
    limit = max(1, min(int(request.args.get('limit', 50)), 100))
    offset = max(0, int(request.args.get('offset', 0)))

    posts = load_community_data()
    if category and category in CATEGORIES:
        posts = [p for p in posts if p.get('category') == category]
    posts = sorted(posts, key=lambda p: p['createdAt'], reverse=True)

    total = len(posts)
    page = posts[offset:offset + limit]
    return jsonify({
        'code': 200,
        'message': '获取成功',
        'data': {
            'items': [_public_post(p) for p in page],
            'total': total,
            'offset': offset,
            'limit': limit,
            'categories': [{'id': k, 'name': v} for k, v in CATEGORIES.items()],
        },
    })


@app.route('/api/community/posts', methods=['POST'])
def community_create_post():
    user, error = require_user()
    if error:
        return error
    data = request.get_json(silent=True) or {}
    title = (data.get('title') or '').strip()
    content = (data.get('content') or '').strip()
    category = data.get('category', 'discussion')
    if category not in CATEGORIES:
        category = 'discussion'

    if not title:
        return jsonify({'code': 400, 'message': '标题不能为空'}), 400
    if not content:
        return jsonify({'code': 400, 'message': '内容不能为空'}), 400
    author = _identity_from_request(data)
    if not author:
        return jsonify({'code': 400, 'message': '请先登录后再发帖'}), 400

    post = {
        'id': str(uuid.uuid4()),
        'title': title,
        'content': content,
        'category': category,
        'author': author,
        'createdAt': datetime.now().isoformat(),
        'replies': [],
    }
    posts = load_community_data()
    posts.append(post)
    save_community_data(posts)

    return jsonify({
        'code': 200,
        'message': '发帖成功',
        'data': _public_post(post),
    })


@app.route('/api/community/posts/<post_id>', methods=['GET'])
def community_get_post(post_id):
    posts = load_community_data()
    for post in posts:
        if post['id'] == post_id:
            return jsonify({
                'code': 200,
                'message': '获取成功',
                'data': {
                    **_public_post(post),
                    'replies': [_public_reply(r) for r in post.get('replies', [])],
                },
            })
    return jsonify({'code': 404, 'message': '帖子不存在'}), 404


@app.route('/api/community/posts/<post_id>/replies', methods=['POST'])
def community_create_reply(post_id):
    user, error = require_user()
    if error:
        return error
    data = request.get_json(silent=True) or {}
    content = (data.get('content') or '').strip()
    if not content:
        return jsonify({'code': 400, 'message': '回复不能为空'}), 400
    author = _identity_from_request(data)
    if not author:
        return jsonify({'code': 400, 'message': '请先登录后再回复'}), 400

    posts = load_community_data()
    for post in posts:
        if post['id'] == post_id:
            reply = {
                'id': str(uuid.uuid4()),
                'content': content,
                'author': author,
                'createdAt': datetime.now().isoformat(),
                'replyTo': data.get('replyTo'),
            }
            post.setdefault('replies', []).append(reply)
            save_community_data(posts)
            return jsonify({
                'code': 200,
                'message': '回复成功',
                'data': _public_reply(reply),
            })
    return jsonify({'code': 404, 'message': '帖子不存在'}), 404


@app.route('/api/community/posts/<post_id>', methods=['DELETE'])
def community_delete_post(post_id):
    user, error = require_user()
    if error:
        return error
    account = user['username']
    posts = load_community_data()
    for i, post in enumerate(posts):
        if post['id'] == post_id:
            if account and account != 'root' and post.get('author', {}).get('account') != account:
                return jsonify({'code': 403, 'message': '无权删除他人帖子'}), 403
            posts.pop(i)
            save_community_data(posts)
            return jsonify({'code': 200, 'message': '删除成功'})
    return jsonify({'code': 404, 'message': '帖子不存在'}), 404


@app.route('/api/community/replies/<reply_id>', methods=['DELETE'])
def community_delete_reply(reply_id):
    user, error = require_user()
    if error:
        return error
    account = user['username']
    posts = load_community_data()
    for post in posts:
        for i, reply in enumerate(post.get('replies', [])):
            if reply['id'] == reply_id:
                if account != 'root' and reply.get('author', {}).get('account') != account:
                    return jsonify({'code': 403, 'message': '无权删除他人回复'}), 403
                post['replies'].pop(i)
                save_community_data(posts)
                return jsonify({'code': 200, 'message': '删除成功'})
    return jsonify({'code': 404, 'message': '回复不存在'}), 404


# ========================================================
# 启动入口
# ========================================================

if __name__ == '__main__':
    print("=" * 50)
    print("唤语算法服务启动中...")
    print(f"上传目录: {UPLOAD_FOLDER}")
    print(f"结果目录: {RESULT_FOLDER}")
    print("TTS 音色列表:")
    for vid, vname in EDGE_TTS_VOICES.items():
        print(f"  - {vid}: {vname}")
    print("=" * 50)
    
    # 调试模式运行，生产环境用 gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)
