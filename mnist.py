import io
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps  # ✅ cv2 不使用で Pillow を活用

# クラスラベル（0〜9）
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
image_size = 28

# Flask アプリの初期化
app = Flask(__name__)

# モデルをロード（compile=False でロード時間短縮）
model = load_model("./model.keras", compile=False)
print("✅ モデルロード完了")
model.summary(print_fn=lambda x: print(x, flush=True))  # モデル構造をログ出力

def preprocess_image(img):
    """ 画像を前処理（余白トリミング、28x28 リサイズ） """
    img = ImageOps.invert(img)  # 白背景・黒文字に変換
    bbox = img.getbbox()
    
    if bbox:
        img = img.crop(bbox)  # 余白をトリミング

    img = img.resize((image_size, image_size), Image.LANCZOS)  # 28x28 にリサイズ
    img = np.array(img, dtype=np.float32) / 255.0  # 0〜1 の範囲に正規化
    return img.reshape(1, image_size, image_size, 1)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    pred_answer = ""

    if request.method == "POST":
        print("📩 POSTリクエスト受信", flush=True)
        print("📂 受け取ったファイル一覧:", request.files.keys(), flush=True)  

        if "file" not in request.files:
            print("⚠️ エラー: ファイルが送信されていません", flush=True)
            return render_template("index.html", answer="ファイルがありません")

        file = request.files["file"]
        print(f"📁 受け取ったファイル名: {file.filename}", flush=True)

        if file.filename == "":
            print("⚠️ エラー: ファイルが選択されていません", flush=True)
            return render_template("index.html", answer="ファイルがありません")

        try:
            # 📸 画像を開く（グレースケールに変換）
            img = Image.open(io.BytesIO(file.read())).convert("L")
            img = preprocess_image(img)  # 前処理（余白トリミング & リサイズ）

            print(f"🖼 画像データの統計: min={img.min()}, max={img.max()}, mean={img.mean()}", flush=True)
            print(f"📏 画像の形状: {img.shape}", flush=True)

            # 🛠 デバッグ用: 前処理後の画像を保存
            plt.imshow(img.reshape(image_size, image_size), cmap="gray")
            plt.title("Processed Image")
            plt.savefig("debug_image.png")  # 画像を保存
            plt.close()

            # 🧠 推論処理
            print("🧪 推論実行中...", flush=True)
            result = model.predict(img)
            print("✅ 推論完了", flush=True)

            if result is None:
                print("⚠️ エラー: `model.predict(img)` の結果が None", flush=True)
                return render_template("index.html", answer="推論に失敗しました")

            print(f"📊 推論結果の型: {type(result)}", flush=True)
            print(f"📊 推論結果の形状: {result.shape}", flush=True)
            print("📊 推論結果の生データ:", result, flush=True)

            # ソフトマックスを適用して確率に変換
            result = tf.nn.softmax(result[0]).numpy()
            print("📊 推論結果の配列:", result, flush=True)

            # 最も確率が高いクラスを予測結果として取得
            predicted = result.argmax()
            pred_answer = f"🔍 きっと、これは {classes[predicted]} じゃないっすか？"
            print("🎯 判定結果:", pred_answer, flush=True)

        except Exception as e:
            print("⚠️ エラー:", e, flush=True)
            return render_template("index.html", answer="エラーが発生しました")

    return render_template("index.html", answer=pred_answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 自動割り当てポートを使用
    print(f"🚀 アプリ起動: ポート {port}", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True)
