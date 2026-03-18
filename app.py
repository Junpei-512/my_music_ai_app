import os
from flask import Flask, render_template, request, url_for
from analyzer import generate_visualizations

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'my_music_ai_app/uploads'
app.config['STATIC_PLOTS'] = 'my_music_ai_app/static/plots'

# 必要なフォルダを自動作成
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_PLOTS'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    plots = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            # ファイルを一時保存
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # 解析実行 (static/plots 内に画像を保存)
            plots = generate_visualizations(file_path, app.config['STATIC_PLOTS'])
            
    return render_template('index.html', plots=plots)

if __name__ == '__main__':
    app.run(debug=True, port=8080)