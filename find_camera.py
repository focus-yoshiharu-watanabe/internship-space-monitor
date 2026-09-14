"""
つながっているカメラの番号を調べる

使い方：python find_camera.py
映ったカメラが1台ずつウィンドウに出るので、使いたいカメラの番号を
app.py の SOURCE に書く。何かキーを押すと次のカメラへ進む。
"""
import sys

import cv2


def open_camera(index):
    if sys.platform == "win32":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


found = []
for index in range(5):
    cap = open_camera(index)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if ok:
        h, w = frame.shape[:2]
        print(f"カメラ {index}: 映りました（{w}x{h}）")
        found.append(index)
        title = f"camera {index} (press any key)"
        cv2.imshow(title, frame)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
    cap.release()

if found:
    print(f"\n使えるカメラの番号: {found}  → app.py の SOURCE に書いてください")
else:
    print("\nカメラが見つかりませんでした。README の「困ったとき」を見てください")
