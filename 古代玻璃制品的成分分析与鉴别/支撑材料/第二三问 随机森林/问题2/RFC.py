'''
训练需要一定时间但不长,主要时间用于网格搜索部分，接近300种模型组合
'''

# 导入必要库(pip install numpy,pandas,matplotlib,sklearn,...)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import metrics
from sklearn.tree import export_graphviz
import os
import pickle
import warnings
warnings.filterwarnings('ignore')



# 预处理已经完成，加载已处理的数据集并划分
data = pd.read_excel('data.xlsx',sheet_name=1)
labels = list(data.columns.values)   
x_data = data.iloc[:,1:16]
y_data = data.iloc[:,16:]
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.1,random_state=6)
x_train = x_train.values
x_test = x_test.values
y_train= y_train.values
y_test = y_test.values


# 超参数（已实验收缩）--网格搜索
'''
下述参数网格为经过逐步缩小网格范围并结合参数特点和问题特点的多次实验的结果
参数范围的选择还参考了最后可视化的结果,防止树太高产生过拟合或者树太宽出现数据利用不充分的状况
采用单一变量的原则，先后缩小：'criterion'、'n_estimators'、'max_depth'、'min_samples_split'、'max_features'
'''
param_grid = {
    'criterion':['entropy','gini'],
    'n_estimators':[11,13,15,17],
    'max_depth':[5, 6, 7],    
    'min_samples_split':[4,8,12,16],
    'max_features':[0.3,0.4,0.5]
}
Rfc_Basic = RandomForestClassifier(random_state = 66)
Rfc_GS = GridSearchCV(estimator=Rfc_Basic, param_grid=param_grid,
                      scoring='accuracy', cv=4)
Rfc_GS.fit(x_train, y_train)

# Rfc_GS.cv_results_
print(Rfc_GS.cv_results_[ 'mean_test_score'])
print(Rfc_GS.cv_results_[ 'std_test_score'])
print('RFC最优模型参数：',Rfc_GS.best_params_)
Rfc_Best=Rfc_GS.best_estimator_
with open('RFC.pickle','wb') as f:
    pickle.dump(Rfc_Best,f)


print('\n--------------------------------------------------')
print('  化学成分\t\t\t\t重要性百分比')
# 最优参数模型各特征对结果的重要性
importances = Rfc_Best.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(15):
    print("%2d)%-*s\t%f" % (i + 1, 30, labels[indices[i]+1], importances[indices[i]]))
print('--------------------------------------------------\n')

pred = Rfc_Best.predict(x_test)
print('测试集中文物真实类型：\t',y_test.tolist())
print('测试集中文物预测类型：\t',pred,'\n')



# 随机森林可视化,可视化代码部分有与 graphviz 的pip install 路径统一的要求，否则可能无法正常运行
fn=['SiO2','Na2O','K2O','CaO','MgO','Al2O3','Fe2O3','CuO','PbO','BaO','P2O5','SrO','SnO2','SO2','weathering']
cn=['lead barium','high potassium']
for index, Tree_estimator in enumerate(Rfc_Best):

    export_graphviz(Tree_estimator,
                    out_file='tree{}.dot'.format(index),
                    feature_names=fn,
                    class_names=cn,
                    rounded=True,
                    proportion=False,
                    precision=2,
                    filled=True)
    os.system('dot -Tpng tree{}.dot -o tree{}.png'.format(index, index))



