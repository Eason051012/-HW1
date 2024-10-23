import numpy as np
import matplotlib.pyplot as plt

def generate_data(a, b, num_points, noise):
    x = np.linspace(0, 10, num_points)
    y = a * x + b + np.random.normal(0, noise, num_points)
    return x, y

def linear_regression(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    return slope, intercept, y_pred

def plot_data(x, y, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(x, y, label='Data Points')
    ax.plot(x, y_pred, color='red', label='Fitted Line')
    ax.legend()
    plt.savefig('static/regression_plot.png')  # 儲存圖片到 static 資料夾
