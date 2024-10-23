import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 讀取 CSV 文件
df = pd.read_csv(r'C:\Users\V52no\OneDrive\文件\Python Scripts\2330-training.csv')

# 假設 CSV 中日期列名為 'Date'，價格列名為 'Price'
# 需要重命名這兩列為 Prophet 期望的 'ds' 和 'y'
df = df.rename(columns={'Date': 'ds', 'Price': 'y'})

# 確保 'ds' 列是日期類型
df['ds'] = pd.to_datetime(df['ds'])

# 移除千位分隔符並將 'y' 列轉換為數字格式
df['y'] = df['y'].replace({',': ''}, regex=True).astype(float)

# 增大 changepoint_prior_scale，增加模型的靈活性
model = Prophet(changepoint_prior_scale=1.0, seasonality_prior_scale=0.1, interval_width=0.7)

# 可以添加自定義季節性（這裡以每月為例）
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# 訓練 Prophet 模型
model.fit(df)

# 生成未來 90 天的日期數據
future = model.make_future_dataframe(periods=90)

# 預測未來的價格
forecast = model.predict(future)

# 查看預測結果
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# 繪製預測結果，並連接原始數據的黑線（無點）
fig, ax = plt.subplots(figsize=(10, 6))

# 繪製 Prophet 預測結果
model.plot(forecast, ax=ax)

# 繪製原始數據，使用黑色線條，去掉黑點
ax.plot(df['ds'], df['y'], color='black', linestyle='-', label='Actual Data')  # 僅連接黑色線條，無點

# 設置標題與標籤
plt.title('Prophet Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Predicted Price')

# 顯示圖例，將圖例移動到圖表內部的左上角
plt.legend(loc='upper left')

# 顯示圖形
plt.show()

# 保存 Prophet 模型
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)
from flask import Flask, request, render_template
import pickle
import pandas as pd
from prophet import Prophet

app = Flask(__name__)

# 加載已保存的 Prophet 模型
with open('prophet_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 獲取用戶輸入的未來日期
    input_date = request.form['date']
    
    # 將日期轉換為 DataFrame 格式以適應 Prophet
    future_date = pd.DataFrame({'ds': [input_date]})
    
    # 使用 Prophet 進行預測
    forecast = model.predict(future_date)
    
    # 返回預測的結果
    predicted_price = forecast['yhat'].values[0]
    
    return render_template('index.html', prediction_text=f'Predicted Price on {input_date}: ${predicted_price:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
