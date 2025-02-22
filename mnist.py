import sys
import subprocess

# OpenCV のインストールチェック
try:
    import cv2
except ModuleNotFoundError:
    print("❌ OpenCV is missing. Attempting to install...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless"])
    import cv2
    print("✅ OpenCV installed successfully:", cv2.__version__)

import os
import logging
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# クラスラベル（0～9）
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
image_size = 28

# 必要なフォルダを作成（存在しない場合）
for folder in ["debug_images", "input_images", "trim-num-file"]:
    os.makedirs(folder, exist_ok=True)

# Flask アプリのセットアップ
app = Flask(__name__)

# 学習済みモデルのロード
model = load_model("./model.keras", compile=False)
logger.info("✅ モデルロード完了")

# 📌 余白削除関数（周囲の不要な余白をカット）
def crop_and_resize(img):
    if img is None:
        logger.error("❌ crop_and_resize: 入力画像が None です！")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # しきい値の調整（50 → 127）
    _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 輪郭を取得
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        logger.error("❌ crop_and_resize: 輪郭が見つかりませんでした！")
        return None

    # 外接矩形を取得
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = gray[y:y+h, x:x+w]

    # リサイズ（28×28）
    resized = cv2.resize(cropped, (28, 28))

    # デバッグ用に保存
    debug_path = os.path.join("debug_images", "cropped_debug.png")
    cv2.imwrite(debug_path, resized)
    logger.info(f"✅ crop_and_resize: Processed image saved to {debug_path}")

    return resized

# 📌 数字を個別に切り出して保存（trim-num-file に格納）
def create_num_file(img):
    _, threshed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_folder = "trim-num-file"
    
    # 既存のファイルを削除
    for file in os.listdir(num_folder):
        os.remove(os.path.join(num_folder, file))

    extracted_numbers = []

    for i, contour in enumerate(sorted(contours, key=lambda c: cv2.boundingRect(c)[0])):  
        x, y, w, h = cv2.boundingRect(contour)
        
        # サイズ制限を緩和（3 → 2）
        if w > 2 and h > 2:
            digit = img[y:y+h, x:x+w]
            resized = cv2.resize(digit, (28, 28))
            save_path = os.path.join(num_folder, f"num{i}.png")
            cv2.imwrite(save_path, resized)
            extracted_numbers.append(save_path)
            logger.info(f"✅ create_num_file: Saved {save_path}")

    if not extracted_numbers:
        logger.error("❌ create_num_file: No digits extracted!")

    return extracted_numbers

@app.route("/", methods=["GET", "POST"])
def upload_file():
    pred_answer = ""

    if request.method == "POST":
        logger.info("📩 POSTリクエスト受信")

        if "file" not in request.files:
            return render_template("index.html", answer="ファイルがありません")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", answer="ファイルがありません")

        # 画像を読み込む
        file_path = os.path.join("input_images", file.filename)
        file.save(file_path)
        img = cv2.imread(file_path)

        # 画像の前処理
        processed_img = crop_and_resize(img)
        if processed_img is None:
            return render_template("index.html", answer="数字が見つかりませんでした")

        # 数字を切り出して保存
        extracted_numbers = create_num_file(processed_img)

        predictions = []

        for num_path in extracted_numbers:
            img = image.load_img(num_path, target_size=(28, 28), color_mode="grayscale")
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            logger.info(f"📸 Processing {num_path}, shape: {img_array.shape}")

            # 推論
            result = model.predict(img_array)
            result = tf.nn.softmax(result[0]).numpy()

            predicted_digit = np.argmax(result)
            predictions.append(classes[predicted_digit])

            logger.info(f"🎯 Prediction: {predicted_digit} ({result})")

        pred_answer = f"🔍 これは {' '.join(predictions)} じゃないっすか？"
        logger.info(f"🎯 判定結果: {pred_answer}")

    return render_template("index.html", answer=pred_answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 アプリ起動: ポート {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
