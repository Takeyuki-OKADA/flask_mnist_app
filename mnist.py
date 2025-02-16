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
print("モデルロード完了")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    pred_answer = ""

    if request.method == 'POST':
        print("POSTリクエスト受信")
        if 'file' not in request.files:
            print("エラー: ファイルがありません")
            return render_template("index.html", answer="ファイルがありません")
        
        file = request.files['file']
        if file.filename == '':
            print("エラー: ファイルが選択されていません")
            return render_template("index.html", answer="ファイルがありません")

        try:
            # ファイルをメモリ上で処理（保存不要）
            img = Image.open(io.BytesIO(file.read())).convert('L').resize((image_size, image_size))
            img = np.array(img).reshape(1, image_size, image_size, 1).astype(np.float32) / 255.0
            
            print(f"画像の形状: {img.shape}")  # ここで形状を確認
            print("推論開始")

            # 推論を実行
            result = model.predict(img)[0]
            print("推論結果の配列:", result)  # ここでモデルの出力を確認

            predicted = result.argmax()
            pred_answer = f"これは {classes[predicted]} です"
            print("判定結果:", pred_answer)  # 判定結果を確認

        except Exception as e:
            print("エラー:", e)  # エラーメッセージを表示
            return render_template("index.html", answer="エラーが発生しました")

    return render_template("index.html", answer=pred_answer)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    print(f"アプリ起動: ポート {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
