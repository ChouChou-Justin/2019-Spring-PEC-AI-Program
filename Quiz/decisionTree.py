from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import pydotplus
import pandas as pd

#pip install graphviz 
#pip install pydotplus 
def drawTree(iris, clf):
    #sklearn.tree.export_graphviz(decision_tree, out_file=None, 
	#max_depth=None, feature_names=None, class_names=None, 
	#label=’all’, filled=False, leaves_parallel=False, 
	#impurity=True, node_ids=False, proportion=False, 
	#rotate=False, rounded=False, special_characters=False, precision=3)
    #https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
    #將 Tree導出為 graphviz
    dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    #使用 pydotplus 產生 pdf 檔案
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('d:\\iris.pdf')

def decisionTree():
    # 讀入鳶尾花資料
    iris = load_iris()
    iris_X = iris.data
    iris_y = iris.target
    print(type(iris.data)) # 資料儲存為 ndarray
    print(iris.feature_names) # 變數名稱可以利用 feature_names 屬性取得
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names) 
	# 轉換為 data frame
    iris_df.loc[:, "species"] = iris.target # 將品種加入 data frame
    
    print(iris_df.head()) # 觀察前五個觀測值    
    #print(iris_df) # 觀察所有觀測值    
    iris_df.hist()  # 劃出直方圖
	#劃出散佈圖
    iris_df.plot.scatter(x='sepal length (cm)', y='sepal width (cm)', c='DarkBlue')
	#Kernel Density Estimates 核心密度估計圖，直方圖的平滑機率分布圖
    iris_df.plot.kde()
    
    print(iris.feature_names)
    
    # 切分訓練與測試資料
    train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)
    #print(train_X)
    
    # 建立分類器
    clf = tree.DecisionTreeClassifier()
	#訓練資料
    iris_clf = clf.fit(train_X, train_y)
    # 預測資料
    test_y_predicted = iris_clf.predict(test_X)
    print(test_y_predicted)
    # 標準答案
    print(test_y)
    # 績效 - 精確度
    accuracy = metrics.accuracy_score(test_y, test_y_predicted)
    print(accuracy)
    drawTree(iris, iris_clf)
    
def testPlot():
    df = pd.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
                   columns=['length', 'width', 'species'])
    df.plot.scatter(x='length', y='width', c='DarkBlue')
    print(df.iloc[1:2])
    
def main():
     testPlot()
     #decisionTree()
     
main()    
