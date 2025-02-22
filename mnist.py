import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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
print("✅ モデルロード完了")

# 📌 余白削除関数（周囲の不要な余白をカット）
def crop_and_resize(img):
    # グレースケール化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二値化（しきい値 = 50）
    _, threshed = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 輪郭を取得
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None  # 数字が見つからない場合

    # 外接矩形を取得（画像の有効領域を決定）
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = gray[y:y+h, x:x+w]
    
    # リサイズ（28×28）
    resized = cv2.resize(cropped, (28, 28))
    
    return resized

# 📌 数字を個別に切り出して保存（trim-num-file に格納）
def create_num_file(img):
    # 二値化
    _, threshed = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 輪郭検出
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 保存フォルダ
    num_folder = "trim-num-file"
    
    # 既存のファイルを削除
    for file in os.listdir(num_folder):
        os.remove(os.path.join(num_folder, file))

    extracted_numbers = []
    
    for i, contour in enumerate(sorted(contours, key=lambda c: cv2.boundingRect(c)[0])):  # X座標でソート
        x, y, w, h = cv2.boundingRect(contour)
        digit = img[y:y+h, x:x+w]  # 数字部分の切り出し
        
        if w > 3 and h > 3:  # 小さすぎるノイズは無視
            resized = cv2.resize(digit, (28, 28))  # 28×28にリサイズ
            save_path = os.path.join(num_folder, f"num{i}.png")
            cv2.imwrite(save_path, resized)  # 保存
            extracted_numbers.append(save_path)

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

        # 画像を読み込む
        file_path = os.path.join("input_images", file.filename)
        file.save(file_path)
        img = cv2.imread(file_path)

        # 画像の前処理（余白削除＆リサイズ）
        processed_img = crop_and_resize(img)
        if processed_img is None:
            return render_template("index.html", answer="数字が見つかりませんでした")

        # デバッグ用に保存
        debug_path = os.path.join("debug_images", file.filename)
        cv2.imwrite(debug_path, processed_img)

        # 個別に数字を切り出して保存
        extracted_numbers = create_num_file(processed_img)

        predictions = []
        
        for num_path in extracted_numbers:
            # 画像を読み込み、モデルに渡す
            img = image.load_img(num_path, target_size=(28, 28), color_mode="grayscale")
            img_array = image.img_to_array(img) / 255.0  # 正規化
            img_array = np.expand_dims(img_array, axis=0)  # バッチ次元を追加
            
            # 推論実行
            result = model.predict(img_array)
            
            # softmax で確率化
            result = tf.nn.softmax(result[0]).numpy()
            
            # 予測結果
            predicted_digit = np.argmax(result)
            predictions.append(classes[predicted_digit])

        pred_answer = f"🔍 これは {' '.join(predictions)} じゃないっすか？"
        print(f"🎯 判定結果: {pred_answer}")

    return render_template("index.html", answer=pred_answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 アプリ起動: ポート {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
