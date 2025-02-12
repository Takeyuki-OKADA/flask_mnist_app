import os
from flask import Flask, request, redirect, render_template, flash, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# クラス定義（0～9の手書き数字）
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Flaskアプリ作成
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "aidemy_secret_key"  # Flaskのセッション用キー（変更可）

# uploadsフォルダが存在しない場合は作成
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 許可された拡張子かどうかをチェックする関数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 学習済みモデルをロード
model = load_model('./model.keras')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # ファイルが送信されているかチェック
        if 'file' not in request.files:
            flash('ファイルが選択されていません', 'error')
            return redirect(request.url)

        file = request.files['file']

        # ファイル名が空かどうかチェック
        if file.filename == '':
            flash('ファイルが選択されていません', 'error')
            return redirect(request.url)

        # 許可されたファイル形式か確認
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 画像を読み込み、モデルが受け取れる形式に変換
            img = image.load_img(filepath, color_mode='grayscale', target_size=(image_size, image_size))
            img = image.img_to_array(img) / 255.0  # 正規化
            data = np.array([img])

            # 予測実行
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = f"これは {classes[predicted]} です"

            return render_template("index.html", answer=pred_answer, image_url=url_for('uploaded_file', filename=filename))

    return render_template("index.html", answer="")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
