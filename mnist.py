import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# クラス（0〜9）
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
image_size = 28  # 画像サイズ

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 学習済みモデルのロード
model = load_model('./model.keras')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    pred_answer = ""

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 画像を読み込み、28x28のグレースケールに変換し、正規化
            img = image.load_img(filepath, color_mode='grayscale', target_size=(image_size, image_size))
            img = image.img_to_array(img) / 255.0  # ★ 画像を0-1に正規化
            data = np.array([img])

            # 予測
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = f"これは {classes[predicted]} です"

    return render_template("index.html", answer=pred_answer)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
