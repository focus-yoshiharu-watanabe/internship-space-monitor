"""
テスト用の動画を撮る

使い方：python record_video.py [カメラ番号] [保存先]
  例）python record_video.py 0 videos/test.mp4
r キーで録画開始／停止、q キーで終了。
撮った動画は app.py の SOURCE に書けば、何度でも同じ映像で試せる。
"""
import sys
from pathlib import Path

import cv2

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "videos/test.mp4")
out_path.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
if not cap.isOpened():
    sys.exit(f"[エラー] カメラ {index} を開けません")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
writer = None
print("r で録画開始／停止、q で終了")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    view = frame.copy()
    if writer:
        writer.write(frame)
        cv2.circle(view, (30, 30), 12, (0, 0, 255), -1)
        cv2.putText(view, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("record (r: start/stop, q: quit)", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        if writer:
            writer.release()
            writer = None
            print(f"保存しました: {out_path}")
        else:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            print("録画中...")
    elif key in (ord("q"), 27):
        break

if writer:
    writer.release()
    print(f"保存しました: {out_path}")
cap.release()
cv2.destroyAllWindows()
