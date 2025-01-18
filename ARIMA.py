import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

# 忽略警告
#warnings.filterwarnings("ignore", category=UserWarning)

# 读取Excel文件
file_path = 'GGDP.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', index_col=0)
print(df)
# 创建一个新的Excel文件来保存预测结果
output_file = 'GGDP_predictions.xlsx'


# 定义一个函数来检查数据平稳性
def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1] <= 0.05  # p-value <= 0.05 表示数据是平稳的


# 定义一个函数来预测未来十年的GGDP
def predict_gdp(series, years=10):
    # 将数据转换为时间序列并设置频率
    series = pd.Series(series.values, index=pd.date_range(start='1970', periods=len(series), freq='YS'))

    # 检查数据平稳性，如果不平稳则进行差分
    if not check_stationarity(series):
        series = series.diff().dropna()  # 一阶差分

    # 拟合ARIMA模型
    model = ARIMA(series, order=(5, 1, 0))  # ARIMA模型的参数可以根据需要调整
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=years)
    return forecast


# 创建一个新的DataFrame来存储预测结果
predictions_df = pd.DataFrame(index=df.index, columns=[f'Predicted_{year}' for year in range(2011, 2021)])

# 对每个国家进行预测
for country in df.index:
    series = df.loc[country].dropna()  # 去除NaN值
    if len(series) > 0:  # 确保有足够的数据进行预测
        forecast = predict_gdp(series)
        predictions_df.loc[country] = forecast

# 将预测结果保存到新的Excel文件中
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    predictions_df.to_excel(writer, sheet_name='Predictions')

print(f"预测结果已保存到 {output_file}")