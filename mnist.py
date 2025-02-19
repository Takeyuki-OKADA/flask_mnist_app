import io
import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# ==============================
# 📌 1. モデルのロード
# ==============================
classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

app = Flask(__name__)

# ✅ モデルをロード
model_path = './svhn_digit_classifier.keras'
model = load_model(model_path, compile=False)
print(f"✅ モデルロード完了: {model_path}")
model.summary(print_fn=lambda x: print(x, flush=True))

# ==============================
# 📌 2. 画像アップロード & 推論
# ==============================
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    pred_answer = ""

    if request.method == 'POST':
        print("📩 POSTリクエスト受信", flush=True)

        if 'file' not in request.files:
            print("❌ エラー: ファイルが送信されていません", flush=True)
            return render_template("index.html", answer="ファイルがありません")

        file = request.files['file']
        if file.filename == '':
            print("❌ エラー: ファイルが選択されていません", flush=True)
            return render_template("index.html", answer="ファイルがありません")

        try:
            img = Image.open(io.BytesIO(file.read())).convert('L').resize((28, 28))
            img = np.array(img).reshape(1, 28, 28, 1).astype(np.float32) / 255.0  # float32 に変更

            print(f"🖼 画像データの統計: min={img.min()}, max={img.max()}, mean={img.mean()}", flush=True)
            print(f"📏 画像の形状: {img.shape}", flush=True)

            # 画像を保存してデバッグ
            plt.imshow(img.reshape(28, 28), cmap="gray")
            plt.title("Processed Image")
            plt.savefig("debug_image.png")
            plt.close()
            
            # 推論処理
            print("🧪 推論実行中...", flush=True)
            result = model.predict(img)
            print("✅ 推論完了", flush=True)

            if result is None:
                print("❌ エラー: `model.predict(img)` の結果が None", flush=True)
                return render_template("index.html", answer="推論に失敗しました")

            print(f"📊 推論結果の型: {type(result)}", flush=True)
            print(f"📊 推論結果の形状: {result.shape}", flush=True)
            print(f"📊 推論結果の生データ: {result}", flush=True)

            # Softmax を適用
            result = tf.nn.softmax(result[0]).numpy()
            print(f"📊 推論結果の配列: {result}", flush=True)

            predicted = result.argmax()
            pred_answer = f"🔍 きっと、これは {classes[predicted]} じゃないっすか？"
            print(f"🎯 判定結果: {pred_answer}", flush=True)

        except Exception as e:
            print("❌ エラー:", e, flush=True)
            return render_template("index.html", answer="エラーが発生しました")

    return render_template("index.html", answer=pred_answer)

# ==============================
# 📌 3. Flask アプリ起動
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 アプリ起動: ポート {port}", flush=True)
    app.run(host='0.0.0.0', port=port, threaded=True)
