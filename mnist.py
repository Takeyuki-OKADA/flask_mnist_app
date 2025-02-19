# ================================
# mnist.py: 手書き数字認識 API (Flask)
# ================================
import os
import io
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image

# ================================
# 必要なライブラリを自動インストール（Render等の環境対応）
# ================================
try:
    import cv2
except ImportError:
    os.system("pip install opencv-python-headless")
    import cv2

# ================================
# 設定 & モデルロード
# ================================
app = Flask(__name__)
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
image_size = 28

# ✅ モデルをロード（compile=False で軽量化）
model_path = "./model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

model = load_model(model_path, compile=False)
print("✅ モデルロード完了")
model.summary(print_fn=lambda x: print(x, flush=True))

# ================================
# Web API
# ================================
@app.route("/", methods=["GET", "POST"])
def upload_file():
    pred_answer = ""

    if request.method == "POST":
        print("✅ POSTリクエスト受信")

        if "file" not in request.files:
            print("❌ エラー: ファイルが送信されていません")
            return render_template("index.html", answer="ファイルがありません")

        file = request.files["file"]
        if file.filename == "":
            print("❌ エラー: ファイルが選択されていません")
            return render_template("index.html", answer="ファイルがありません")

        try:
            # 画像読み込み & 前処理
            img = Image.open(io.BytesIO(file.read())).convert("L")  # グレースケール化
            img = np.array(img)
            
            # ✅ 自動トリミング（余白削除）
            _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(binary_img)
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y+h, x:x+w]

            # ✅ リサイズ & 正規化
            img = cv2.resize(img, (28, 28))
            img = img.astype(np.float32) / 255.0
            img = img.reshape(1, 28, 28, 1)

            # ✅ 推論実行
            result = model.predict(img)
            predicted = np.argmax(result)

            pred_answer = f"きっと、これは {classes[predicted]} じゃないっすか？"
            print(f"✅ 判定結果: {pred_answer}")

        except Exception as e:
            print("❌ エラー:", e)
            return render_template("index.html", answer="エラーが発生しました")

    return render_template("index.html", answer=pred_answer)

# ================================
# アプリ起動
# ================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 アプリ起動: ポート {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
