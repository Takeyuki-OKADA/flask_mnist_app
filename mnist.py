import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# クラスラベル
classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

# アップロードフォルダの設定
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Flask アプリの設定
app = Flask(__name__)

# **メモリ対策: アップロードファイルサイズを2MB以下に制限**
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB

# **学習済みモデルのロード**
model = load_model('./model.keras')

# 許可された拡張子かどうかをチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
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
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # **画像を読み込み、np.array に変換（メモリ対策: float16を使用）**
            img = image.load_img(filepath, color_mode='grayscale', target_size=(image_size, image_size))
            img = image.img_to_array(img, dtype=np.float16)  # **float16 に変更**
            data = np.array([img])

            # **推論の実行**
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = f"これは {classes[predicted]} です"

            return render_template("index.html", answer=pred_answer)

    return render_template("index.html", answer="")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
