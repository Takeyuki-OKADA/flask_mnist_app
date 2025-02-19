# ================================
# mnist.py: デバッグ強化版 (Flask)
# ================================
import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import matplotlib.pyplot as plt

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
            # ✅ 画像読み込み & 前処理
            img = Image.open(io.BytesIO(file.read())).convert("L")  # グレースケール化
            img = np.array(img)

            # ✅ 受け取った画像を保存（デバッグ用）
            plt.imshow(img, cmap="gray")
            plt.title("Original Image")
            plt.savefig("debug_input.png")
            plt.close()
            print("✅ 受信画像を debug_input.png に保存")

            # ✅ 画像の統計情報（デバッグ用）
            print(f"画像データの統計: min={img.min()}, max={img.max()}, mean={img.mean()}, shape={img.shape}")

            # ✅ 自動トリミング（余白削除）
            _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(binary_img)
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y+h, x:x+w]

            # ✅ トリミング後の画像を保存（デバッグ用）
            plt.imshow(img, cmap="gray")
            plt.title("Trimmed Image")
            plt.savefig("debug_trimmed.png")
            plt.close()
            print("✅ トリミング後の画像を debug_trimmed.png に保存")

            # ✅ リサイズ & 正規化
            img = cv2.resize(img, (28, 28))
            img = img.astype(np.float32) / 255.0
            img = img.reshape(1, 28, 28, 1)

            # ✅ リサイズ後の画像を保存（デバッグ用）
            plt.imshow(img.reshape(28, 28), cmap="gray")
            plt.title("Resized Image")
            plt.savefig("debug_resized.png")
            plt.close()
            print("✅ リサイズ後の画像を debug_resized.png に保存")

            # ✅ NumPy 配列の統計情報（デバッグ用）
            print(f"リサイズ後の画像データ: min={img.min()}, max={img.max()}, mean={img.mean()}, shape={img.shape}")

            # ✅ 推論実行
            print("🔍 推論開始...")
            result = model.predict(img)
            print("✅ 推論完了")

            # ✅ 推論結果（各クラスの確率）
            print("🔍 推論結果（確率）:", result)

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
