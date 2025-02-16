import io
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

# Flaskアプリケーションのインスタンスを定義
app = Flask(__name__)

# モデルを起動時に読み込む（compile=False でロード時間短縮）
model = load_model('./model.keras', compile=False)
print("モデルロード完了")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    pred_answer = ""

    if request.method == 'POST':
        print("POSTリクエスト受信")
        print("受け取ったファイル一覧:", request.files.keys())  # デバッグ用
        
        if 'file' not in request.files:
            print("エラー: ファイルが送信されていません")
            return render_template("index.html", answer="ファイルがありません")

        file = request.files['file']
        print("受け取ったファイル名:", file.filename)  # デバッグ用

        if file.filename == '':
            print("エラー: ファイルが選択されていません")
            return render_template("index.html", answer="ファイルがありません")

        try:
            img = Image.open(io.BytesIO(file.read())).convert('L').resize((28, 28))
            img = np.array(img).reshape(1, 28, 28, 1).astype(np.float32) / 255.0  # float16 → float32 に変更

            print(f"画像の形状: {img.shape}")
            print("推論直前のメモリ状態確認")

            # 画像を確認（デバッグ用）
            plt.imshow(img.reshape(28, 28), cmap="gray")
            plt.title("Processed Image")
            plt.savefig("debug_image.png")  # 画像を保存
            plt.close()
            
            # 推論時間の測定
            start_time = time.time()
            result = model.predict(img)[0]
            result = tf.nn.softmax(result).numpy()  # softmax を適用
            end_time = time.time()
            print(f"推論時間: {end_time - start_time:.2f} 秒")

            print("推論結果の配列:", result)

            predicted = result.argmax()
            pred_answer = f"これは {classes[predicted]} です"
            print("判定結果:", pred_answer)

        except Exception as e:
            print("エラー:", e)
            return render_template("index.html", answer="エラーが発生しました")

    return render_template("index.html", answer=pred_answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render の自動割り当てポートを使用
    print(f"アプリ起動: ポート {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)