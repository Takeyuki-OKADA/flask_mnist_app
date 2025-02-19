import io
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image

# クラス定義（0～9）
classes = ["0","1","2","3","4","5","6","7","8","9"]

# 画像サイズ（28x28 or 32x32）
image_size = 28  # 変更する場合は 32 に

# Flaskアプリケーションのインスタンスを定義
app = Flask(__name__)

# モデルをロード
model = load_model('./model.keras', compile=False)
print("✅ モデルロード完了")
model.summary(print_fn=lambda x: print(x, flush=True))  # モデルの構造をログ出力

def preprocess_image(img):
    """ スマホで撮影した手書き数字をトリミング & リサイズして前処理 """
    
    # OpenCV 形式に変換
    img = np.array(img)

    # しきい値処理（バイナリ化）
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # 輪郭を検出
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最も大きい輪郭を取得（数字の領域）
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img_cropped = img[y:y+h, x:x+w]  # トリミング
    else:
        img_cropped = img  # 輪郭がなければそのまま

    # 目的のサイズにリサイズ（28x28 or 32x32）
    img_resized = cv2.resize(img_cropped, (image_size, image_size))

    # 形状を (28, 28, 1) にして正規化
    img_resized = img_resized.reshape(image_size, image_size, 1).astype(np.float32) / 255.0

    return img_resized

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    pred_answer = ""

    if request.method == 'POST':
        print("📥 POSTリクエスト受信", flush=True)
        print("受け取ったファイル一覧:", request.files.keys(), flush=True)  # デバッグ用
        
        if 'file' not in request.files:
            print("❌ エラー: ファイルが送信されていません", flush=True)
            return render_template("index.html", answer="ファイルがありません")

        file = request.files['file']
        print("📂 受け取ったファイル名:", file.filename, flush=True)  # デバッグ用

        if file.filename == '':
            print("❌ エラー: ファイルが選択されていません", flush=True)
            return render_template("index.html", answer="ファイルがありません")

        try:
            # 画像を開く & 前処理
            img = Image.open(io.BytesIO(file.read())).convert('L')  # グレースケール化
            img = preprocess_image(img)  # 前処理（トリミング & リサイズ）

            # 形状を (1, 28, 28, 1) に変換
            img = np.expand_dims(img, axis=0)

            print(f"🖼️ 画像データの統計: min={img.min()}, max={img.max()}, mean={img.mean()}", flush=True)
            print(f"📏 画像の形状: {img.shape}", flush=True)
            
            # 画像を確認（デバッグ用）
            plt.imshow(img.reshape(image_size, image_size), cmap="gray")
            plt.title("Processed Image")
            plt.savefig("debug_image.png")  # 画像を保存
            plt.close()
            
            # 推論処理
            print("🔍 推論実行中...", flush=True)
            try:
                result = model.predict(img)
                print("✅ 推論が完了しました", flush=True)

                if result is None:
                    print("❌ エラー: `model.predict(img)` の結果が None です", flush=True)
                    return render_template("index.html", answer="推論に失敗しました")
                
                print(f"📊 推論結果の型: {type(result)}", flush=True)  
                print(f"📊 推論結果の形状: {result.shape}", flush=True)  
                print("📊 推論結果の生データ:", result, flush=True)

                # softmax を適用して確率に変換
                result = tf.nn.softmax(result[0]).numpy()
                print("📊 推論結果の配列:", result, flush=True)

                predicted = result.argmax()
                pred_answer = f"✨ きっと、これは {classes[predicted]} じゃないっすか？"
                print("✅ 判定結果:", pred_answer, flush=True)

            except Exception as e:
                print("❌ エラー: 推論処理中に例外発生", e, flush=True)
                return render_template("index.html", answer="推論に失敗しました")

        except Exception as e:
            print("❌ エラー:", e, flush=True)
            return render_template("index.html", answer="エラーが発生しました")

    return render_template("index.html", answer=pred_answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render の自動割り当てポートを使用
    print(f"🚀 アプリ起動: ポート {port}", flush=True)
    app.run(host='0.0.0.0', port=port, threaded=True)
