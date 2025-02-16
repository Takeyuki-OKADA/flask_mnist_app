import io
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

app = Flask(__name__)

# モデルを起動時に読み込む（compile=False でロード時間短縮）
model = load_model('./model.keras', compile=False)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    pred_answer = ""

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("index.html", answer="ファイルがありません")
        
        file = request.files['file']
        if file.filename == '':
            return render_template("index.html", answer="ファイルがありません")

        # ファイルをメモリ上で処理（保存不要）
        img = Image.open(io.BytesIO(file.read())).convert('L').resize((image_size, image_size))
        img = np.array(img).reshape(1, image_size, image_size, 1).astype(np.float16)

        # 推論を実行
        result = model.predict(img)[0]
        predicted = result.argmax()
        pred_answer = f"これは {classes[predicted]} です"

    return render_template("index.html", answer=pred_answer)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)
