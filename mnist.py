@app.route('/', methods=['GET', 'POST'])
def upload_file():
    pred_answer = ""

    if request.method == 'POST':
        print("POSTリクエスト受信")
        print("受け取ったファイル一覧:", request.files.keys())  # 追加

        if 'file' not in request.files:
            print("エラー: ファイルが送信されていません")
            return render_template("index.html", answer="ファイルがありません")

        file = request.files['file']
        if file.filename == '':
            print("エラー: ファイルが選択されていません")
            return render_template("index.html", answer="ファイルがありません")

        try:
            img = Image.open(io.BytesIO(file.read())).convert('L').resize((28, 28))
            img = np.array(img).reshape(1, 28, 28, 1).astype(np.float32) / 255.0

            print(f"画像の形状: {img.shape}")
            print("推論開始")

            result = model.predict(img)[0]
            print("推論結果の配列:", result)

            predicted = result.argmax()
            pred_answer = f"これは {classes[predicted]} です"
            print("判定結果:", pred_answer)

        except Exception as e:
            print("エラー:", e)
            return render_template("index.html", answer="エラーが発生しました")

    return render_template("index.html", answer=pred_answer)
