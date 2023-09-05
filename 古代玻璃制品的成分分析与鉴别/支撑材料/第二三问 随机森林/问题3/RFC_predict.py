import pickle
import pandas as pd

#导入模型（来自问题2的随机森林）
with open('RFC.pickle','rb') as f:
    Rfc_Best = pickle.load(f)

# 问题3预测
data_pred = pd.read_excel('data.xlsx',sheet_name=2)
x_pred = data_pred.iloc[:,2:17].values
result = Rfc_Best.predict(x_pred)
print('RandomForestClassifier 问题3文物类型预测结果：',result)

