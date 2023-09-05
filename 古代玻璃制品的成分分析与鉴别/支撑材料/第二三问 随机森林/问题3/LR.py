from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

scaler = MinMaxScaler()

# 加载并预处理数据集
data = pd.read_excel('data.xlsx',sheet_name=1)
labels = list(data.columns.values)
x_data = data.iloc[:,1:16]
y_data = data.iloc[:,16:]
x = x_data.values
x= scaler.fit_transform(x)
y = y_data.values
model = LogisticRegression()
model.fit(x,y)

#问题3预测
data_pred = pd.read_excel('data.xlsx',sheet_name=2)
x_pred = data_pred.iloc[:,2:17].values
x_pred= scaler.fit_transform(x_pred)
result = model.predict(x_pred)
print('问题3文物类型预测结果：',result)