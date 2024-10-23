import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

app = Flask(__name__)

# 預測函數：處理自動回歸或多重線性回歸，並生成圖表
def run_regression(changepoint_prior_scale, seasonality_prior_scale, custom_changepoints=None):
    # 讀取 CSV 文件
    df = pd.read_csv('2330-training.csv')  # 確保路徑正確

    # 整理數據
    df = df.rename(columns={'Date': 'ds', 'Price': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['y'].replace({',': ''}, regex=True).astype(float)

    # 初始化 Prophet 模型，根據用戶設置調整參數
    model = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale)
    
    # 如果設置了手動變化點，則應用這些變化點
    if custom_changepoints:
        model.changepoints = custom_changepoints

    # 添加自定義季節性（每年、每季度等），如果需要
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    # 訓練 Prophet 模型
    model.fit(df)

    # 生成未來 365 天的日期數據
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # 繪製預測結果，並保存圖表
    fig, ax = plt.subplots(figsize=(10, 6))
    model.plot(forecast, ax=ax)
    plt.legend(loc='upper left')
    plt.savefig('static/forecast_plot.png')
    plt.close()

# Flask 路由：主頁面
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 獲取用戶輸入的模型參數
        changepoint_prior_scale = float(request.form.get('changepoint_prior_scale', 0.05))
        seasonality_prior_scale = float(request.form.get('seasonality_prior_scale', 10))
        custom_changepoints = request.form.get('custom_changepoints')

        # 如果用戶輸入了自定義變化點，將其轉換為日期列表
        if custom_changepoints:
            custom_changepoints = [pd.to_datetime(cp) for cp in custom_changepoints.split(',')]

        # 執行模型預測
        run_regression(changepoint_prior_scale, seasonality_prior_scale, custom_changepoints)
        return redirect(url_for('result'))
    return render_template('index.html')

# 結果頁面：顯示預測圖表
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
