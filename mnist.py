import os
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

# クラスラベル
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
IMAGE_SIZE = 28

# ディレクトリ構成を確認
os.makedirs("debug_images", exist_ok=True)
os.makedirs("input_images", exist_ok=True)
os.makedirs("trim-num-file", exist_ok=True)

# Flaskアプリの初期化
app = Flask(__name__)

# 学習済みモデルをロード
MODEL_PATH = "./model.keras"
model = load_model(MODEL_PATH, compile=False)
print("✅ モデルロード完了")
model.summary()

def preprocess_image(img):
    """
    受け取った画像を前処理する
    - 余白削除
    - 二値化
    - ノイズ処理
    - 28x28 にリサイズ
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # ノイズ除去 & 文字の太さ補正
    kernel = np.ones((2,2), np.uint8)
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    
    # 輪郭抽出 → 余白をカット
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img_bin = img_bin[y:y+h, x:x+w]
    
    # 28x28 にリサイズ
    img_resized = cv2.resize(img_bin, (IMAGE_SIZE, IMAGE_SIZE))
    img_resized = img_resized.astype(np.float32) / 255.0  # 正規化
    img_resized = np.expand_dims(img_resized, axis=(0, -1))  # (1, 28, 28, 1)
    
    return img_resized

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("index.html", answer="ファイルがありません")
        
        file = request.files['file']
        if file.filename == '':
            return render_template("index.html", answer="ファイルがありません")
        
        try:
            # 画像を読み込み
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            img = np.array(img)
            
            # 前処理
            img_processed = preprocess_image(img)
            
            # 推論
            prediction_probs = model.predict(img_processed)[0]
            predicted_label = np.argmax(prediction_probs)
            prediction = f"🔍 これは {CLASSES[predicted_label]} です"
            
            # デバッグ用ログ保存
            plt.imshow(img_processed[0, :, :, 0], cmap='gray')
            plt.title(f"Pred: {CLASSES[predicted_label]}")
            plt.savefig(f"debug_images/{file.filename}.png")
            plt.close()
            
        except Exception as e:
            print("エラー:", e)
            return render_template("index.html", answer="エラーが発生しました")
    
    return render_template("index.html", answer=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
