
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')




# 加载数据集
data = pd.read_excel('data.xlsx',sheet_name=1)
labels = list(data.columns.values)
x_data = data.iloc[:,1:16]
y_data = data.iloc[:,16:17]
x = x_data.values
y = y_data.values

def select_best_tree(x,y):
    k_fold = KFold(n_splits=10)
    tree_model = DecisionTreeClassifier(random_state=20)
    params = {'max_depth':range(1,16),'criterion':np.array(['entropy','gini'])}
    grid = GridSearchCV(tree_model, param_grid=params,scoring='neg_mean_squared_error',cv=k_fold)
    grid = grid.fit(x, y)
    return grid.best_estimator_


tree_model=select_best_tree(x,y)

tree_model.fit(x,y)
scores=cross_val_score(tree_model, x, y, cv=5,scoring = "precision_weighted")
print("out:{} mean: {:.3f} (std: {:.3f})".format(scores,scores.mean(),scores.std()),end="\n" )

fn=labels[1:16]
cn=['铅钡','高钾']

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(tree_model,
               feature_names = fn,
               class_names=cn,
               filled = True);
#fig.savefig('imagename.png')

pre_data = pd.read_excel('data.xlsx',sheet_name=2)
pre_x = pre_data.iloc[:,2:17].values
pre_res=tree_model.predict(pre_x)
print(pre_res)

