"""
スペース見守りモニター（土台）

  カメラ映像 → AI（YOLOX）で人を検出 → count_people で人数を数える
  → judge_xxx でルールを判定 → 画面に表示

参加者が触ってよいのは、下の「設定」欄だけです。
"""
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detector import COCO_CLASSES, Detector
from common import count_people
from capacity import judge_capacity
from absence import judge_absence

# ============================================================
# 設定（ここは自由に変えてOK）
# ============================================================

# 映像。Jetson では担当者が練習用の動画を2本書いてある。使う方の先頭の # を外し、もう片方に # をつける
# （PC で動かすときはカメラ番号 0, 1, 2 ... や "videos/test.mp4" のような動画のパス）
SOURCE = 0

# 検知するもの（例：["person", "cup", "cell phone"]）。英語の名前で書く
CLASSES = ["person"]

# AIの自信がこれより低い検出は無視する（0.0〜1.0）
CONF = 0.5

# 見守るゾーン (左上x, 左上y, 右下x, 右下y)。単位はピクセル
# 画面は横幅 640 に縮めて表示している。マウスを乗せると座標が右下に出る
ZONE = (160, 60, 480, 340)

# 動かすルール。自分の担当のものだけ残す。統合するときは両方並べる
RULES = [judge_capacity]
# RULES = [judge_absence]
# RULES = [judge_capacity, judge_absence]

# Alert の level ごとの色 (赤, 緑, 青)。2人で決めた約束に合わせて書き換える
LEVEL_COLORS = {
    "warn": (230, 40, 40),   # 赤
    "info": (240, 190, 0),   # 黄
}

# ============================================================
# ここから下は土台です（触らないでください）
# ============================================================

MODEL = Path(__file__).resolve().parent / "models" / "yolox_tiny.onnx"
FRAME_WIDTH = 640
# 日本語にすると Jetson（Qt）でウィンドウが開けない。Jetson で2人の画面を見分けるためユーザー名を入れる
WINDOW = " ".join(filter(None, ["Space Monitor", os.environ.get("USER"), "(q: quit)"]))
WINDOW_SIZE = (960, 540)  # 1920x1080 のモニタに2人分並ぶ大きさ
ALERT_KEYS = {"rule", "level", "message"}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    print("[注意] 日本語フォントが見つからないため、日本語が□で表示されます")
    return ImageFont.load_default(size)


def open_source(source):
    """カメラ（番号か /dev/... ）か動画ファイルを開く。(cap, 動画ファイルかどうか) を返す"""
    if isinstance(source, int) or str(source).startswith("/dev/"):
        if sys.platform == "win32":
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            sys.exit(f"[エラー] カメラ {source} を開けません。"
                     "ほかのプログラムが使っていないか、ケーブルが抜けていないか確認してください")
        return cap, False

    if not Path(source).exists():
        sys.exit(f"[エラー] 動画ファイル {source} が見つかりません")
    return cv2.VideoCapture(str(source)), True


def class_ids(names):
    unknown = set(names) - set(COCO_CLASSES)
    if unknown:
        sys.exit(f"[エラー] CLASSES に知らない名前があります: {sorted(unknown)}\n"
                 f"使える名前: {sorted(COCO_CLASSES)}")
    return [i for i, name in enumerate(COCO_CLASSES) if name in names]


def detect(detector, frame, ids):
    """AIで検出して、参加者に渡す形（辞書のリスト）に変換する"""
    detections = []
    for x1, y1, x2, y2, score, cls in detector(frame, CONF, ids):
        detections.append({
            "class": COCO_CLASSES[cls],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": round(score, 2),
        })
    return detections


_reported_errors = set()


def report_error(where):
    """参加者のコードで起きた例外を、同じものは1回だけターミナルに出す"""
    detail = traceback.format_exc()
    if detail not in _reported_errors:
        _reported_errors.add(detail)
        print(f"\n===== {where} でエラー =====\n{detail}")
    return f"{where} でエラー（詳細はターミナル）"


def run_count(detections):
    try:
        count = count_people(detections, ZONE)
    except Exception:
        return 0, report_error("common.py の count_people")
    if not isinstance(count, int):
        return 0, f"count_people の返り値が整数ではありません: {count!r}"
    return count, None


def run_rule(rule, count, now, state):
    """ルールを1つ動かす。約束と違う返り値は、画面で分かるように変換する"""
    name = rule.__name__
    try:
        alert = rule(count, now, state)
    except Exception:
        return {"rule": name, "level": "_error", "message": report_error(name)}
    if alert is None:
        return None
    if not isinstance(alert, dict) or not ALERT_KEYS <= alert.keys():
        return {"rule": name, "level": "_error",
                "message": f"{name} の返り値が約束の形ではありません: {alert!r}"}
    return alert


def draw(frame, detections, count, count_error, alerts, fps, mouse, font, big_font):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle(ZONE, fill=(60, 140, 255, 50), outline=(60, 140, 255, 255), width=3)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(image)

    for det in detections:
        d.rectangle((det["x1"], det["y1"], det["x2"], det["y2"]), outline=(0, 220, 0), width=2)
        d.text((det["x1"] + 3, det["y1"] + 2), f'{det["class"]} {det["conf"]}', fill=(0, 220, 0), font=font)

    w, h = image.size
    d.rectangle((0, 0, 150, 58), fill=(0, 0, 0))
    d.text((8, 2), f"現在 {count}人", fill=(255, 255, 255), font=big_font)
    d.text((8, 38), f"FPS {fps:.1f}", fill=(180, 180, 180), font=font)
    if count_error:
        d.text((160, 8), count_error, fill=(255, 80, 80), font=font)

    y = h - 8
    for alert in reversed(alerts):
        level = alert["level"]
        color = LEVEL_COLORS.get(level)
        message = str(alert["message"])
        if color is None:
            color = (255, 0, 255) if level != "_error" else (255, 80, 80)
            if level != "_error":
                message += f"（level '{level}' は LEVEL_COLORS にない）"
        d.rectangle((0, 0, w - 1, h - 1), outline=color, width=10)
        text_font = font if level == "_error" else big_font
        box = d.textbbox((0, 0), message, font=text_font)
        y -= box[3] + 10
        d.rectangle((10, y - 4, 20 + box[2], y + box[3] + 6), fill=(0, 0, 0))
        d.text((15, y), message, fill=color, font=text_font)

    if mouse:
        label = f"x={mouse[0]} y={mouse[1]}"
        box = d.textbbox((0, 0), label, font=font)
        d.text((w - box[2] - 12, h - box[3] - 12), label, fill=(255, 255, 0), font=font)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def resize(frame):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))


def parse_source():
    """python app.py 1 や python app.py videos/test.mp4 で SOURCE を上書きできる"""
    if len(sys.argv) < 2:
        return SOURCE
    arg = sys.argv[1]
    return int(arg) if arg.isdigit() else arg


def main():
    source = parse_source()
    print("AIモデルを読み込んでいます...")
    detector = Detector(MODEL)
    print(f"推論に使う装置: {detector.device}")
    ids = class_ids(CLASSES)
    cap, is_video = open_source(source)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    font, big_font = load_font(16), load_font(26)
    states = {rule: {} for rule in RULES}
    mouse = []

    def on_mouse(event, x, y, flags, param):
        mouse[:] = [x, y]

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW, *WINDOW_SIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    print("起動しました。ウィンドウで q キー、またはこのターミナルで Ctrl+C を押すと終了します")
    try:
        loop(detector, ids, cap, is_video, video_fps, states, mouse, font, big_font)
    except KeyboardInterrupt:
        pass
    cap.release()
    cv2.destroyAllWindows()
    print("終了しました")


def loop(detector, ids, cap, is_video, video_fps, states, mouse, font, big_font):
    loop_offset = 0.0   # 動画を巻き戻したとき、時刻が戻らないようにする
    last_video_time = 0.0
    prev = time.time()
    fps = 0.0

    while True:
        started = time.time()
        ok, frame = cap.read()
        if not ok:
            if is_video:
                loop_offset += last_video_time
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            sys.exit("[エラー] カメラから映像が来ません。ケーブルが抜けていないか確認してください")

        if is_video:
            last_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            now = loop_offset + last_video_time
        else:
            now = time.time()

        frame = resize(frame)
        detections = detect(detector, frame, ids)
        count, count_error = run_count(detections)
        alerts = [a for a in (run_rule(r, count, now, states[r]) for r in RULES) if a]

        fps = 0.9 * fps + 0.1 * (1 / max(started - prev, 1e-6))
        prev = started
        cv2.imshow(WINDOW, draw(frame, detections, count, count_error, alerts, fps, mouse, font, big_font))

        wait_ms = 1
        if is_video:  # 動画は実際の速さで再生する
            wait_ms = max(1, int(1000 / video_fps - (time.time() - started) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break


if __name__ == "__main__":
    main()
