import sys
import subprocess
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# OpenCV のインストール確認
try:
    import cv2
    print("✅ OpenCV is successfully imported:", cv2.__version__)
except ModuleNotFoundError:
    print("❌ OpenCV is missing. Attempting to install...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless"])
    import cv2
    print("✅ OpenCV installed successfully:", cv2.__version__)

# クラスラベル（0～9）
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
image_size = 28

# 必要なフォルダを作成
for folder in ["debug_images", "input_images", "trim-num-file"]:
    os.makedirs(folder, exist_ok=True)

# Flask アプリのセットアップ
app = Flask(__name__)

# 学習済みモデルのロード
model = load_model("./model.keras", compile=False)
print("✅ モデルロード完了")

# 📌 余白削除関数

def crop_and_resize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("⚠ 数字が見つかりませんでした（輪郭なし）")
        return None

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = gray[y:y+h, x:x+w]
    debug_path = os.path.join("debug_images", "cropped_debug.png")
    cv2.imwrite(debug_path, cropped)
    print(f"✅ 余白削除後の画像を保存: {debug_path}")

    resized = cv2.resize(cropped, (28, 28))
    return resized

# 📌 数字を切り出して保存

def create_num_file(img):
    _, threshed = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_folder = "trim-num-file"
    for file in os.listdir(num_folder):
        os.remove(os.path.join(num_folder, file))

    extracted_numbers = []
    for i, contour in enumerate(sorted(contours, key=lambda c: cv2.boundingRect(c)[0])):
        x, y, w, h = cv2.boundingRect(contour)
        digit = img[y:y+h, x:x+w]
        if w > 3 and h > 3:
            resized = cv2.resize(digit, (28, 28))
            save_path = os.path.join(num_folder, f"num{i}.png")
            cv2.imwrite(save_path, resized)
            extracted_numbers.append(save_path)

    if not extracted_numbers:
        print("⚠ 数字が検出されませんでした（輪郭なし）")
    else:
        print(f"✅ {len(extracted_numbers)} 個の数字を切り出しました")

    return extracted_numbers

@app.route("/", methods=["GET", "POST"])
def upload_file():
    pred_answer = ""
    if request.method == "POST":
        print("📩 POSTリクエスト受信")
        if "file" not in request.files:
            return render_template("index.html", answer="ファイルがありません")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", answer="ファイルがありません")
        
        file_path = os.path.join("input_images", file.filename)
        file.save(file_path)
        img = cv2.imread(file_path)
        processed_img = crop_and_resize(img)
        if processed_img is None:
            return render_template("index.html", answer="数字が見つかりませんでした")
        
        debug_path = os.path.join("debug_images", file.filename)
        cv2.imwrite(debug_path, processed_img)
        extracted_numbers = create_num_file(processed_img)

        predictions = []
        for num_path in extracted_numbers:
            img = image.load_img(num_path, target_size=(28, 28), color_mode="grayscale")
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            print(f"🔍 {num_path} をモデルに入力")
            result = model.predict(img_array)
            result = tf.nn.softmax(result[0]).numpy()
            predicted_digit = np.argmax(result)
            confidence = result[predicted_digit]
            print(f"🎯 予測: {predicted_digit}（信頼度: {confidence:.4f}）")
            predictions.append(classes[predicted_digit])

        pred_answer = f"🔍 これは {' '.join(predictions)} じゃないっすか？" if predictions else "⚠ 画像から数字が認識できませんでした"
    
    return render_template("index.html", answer=pred_answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 アプリ起動: ポート {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
