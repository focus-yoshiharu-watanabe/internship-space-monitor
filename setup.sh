#!/usr/bin/env bash
# Jetson（Linux）用セットアップ
set -e
cd "$(dirname "$0")"

echo "=== 1/3 Python の仮想環境を作ります ==="
python3 -m venv .venv

echo "=== 2/3 ライブラリを入れます（数分かかります） ==="
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# pip 版 OpenCV の Qt がフォントを探して警告を出し続けるので、システムのフォントを見せる
for qt in .venv/lib/python3*/site-packages/cv2/qt; do
  [ -d "$qt" ] && [ ! -e "$qt/fonts" ] && ln -s /usr/share/fonts/truetype/dejavu "$qt/fonts" || true
done

echo "=== 3/3 AIモデルを確認します ==="
.venv/bin/python -c "from detector import Detector; print('model OK:', Detector('models/yolox_tiny.onnx').device)"

echo
echo "完了しました。次は ./run.sh で起動してください。"
