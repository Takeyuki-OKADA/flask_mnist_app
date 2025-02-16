import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = "supersecretkey"  # セッションを使用するために設定

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model.keras')  # 学習済みモデルをロード

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    answer = ""
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
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # 画像を適切に処理する
            img = image.load_img(file, color_mode='grayscale', target_size=(image_size, image_size))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            # 予測処理
            result = model.predict(img)[0]
            predicted = result.argmax()
            answer = f"これは {classes[predicted]} です"
            print(answer)  # デバッグ用

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))  # Renderのポートに合わせる
    app.run(host='0.0.0.0', port=port)
