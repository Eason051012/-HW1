from flask import Flask, render_template, request
import os
from regression import generate_data, linear_regression, plot_data

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    slope, intercept = None, None  # 預設的回歸結果為 None
    if request.method == 'POST':
        # 從表單中獲取用戶輸入的參數
        a = float(request.form['a'])
        b = float(request.form['b'])
        num_points = int(request.form['num_points'])
        noise = float(request.form['noise'])

        # 生成數據和進行線性回歸
        x, y = generate_data(a, b, num_points, noise)
        slope, intercept, y_pred = linear_regression(x, y)
        plot_data(x, y, y_pred)

    # 無論是 GET 還是 POST，最終都渲染 index.html
    return render_template('index.html', slope=slope, intercept=intercept)

if __name__ == '__main__':
    # 確保靜態文件夾存在
    os.makedirs(os.path.join(app.instance_path, 'static'), exist_ok=True)
    app.run(debug=True)
