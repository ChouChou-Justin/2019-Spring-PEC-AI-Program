import pandas as pd 
import seaborn as sns
import matplotlib as plt 
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from yellowbrick.classifier import ClassificationReport 
sales_data = pd.read_csv("C:\\Users\\NTUTCSIE\\Downloads\\WA_Fn-UseC_-Sales-Win-Loss.csv")
#pd.set_option('display.max_columns',None)
#print(sales_data)
#print(sales_data.head(n=2))
#print(sales_data.tail(n=2))
#print(sales_data.dtypes)
#sns.countplot('Route To Market', data=sales_data,hue='Opportunity Result')
#sns.violinplot(x='Opportunity Result',y='Client Size By Revenue',hue='Opportunity Result',data=sales_data)
le = preprocessing.LabelEncoder()
encoded_value = le.fit_transform(["paris","paris","tokyo","amsterdam"])
#print(encoded_value)
sales_data['Supplies Subgroup']=le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region']=le.fit_transform(sales_data['Region'])
sales_data['Route To Market']=le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result']=le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type']=le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group']=le.fit_transform(sales_data['Supplies Group'])
#print(sales_data.head())
cols = [col for col in sales_data.columns if col not in ['Opportunity Number','Opportunity Result']]
data = sales_data[cols]
target = sales_data['Opportunity Result']
#print(data.head(n=2))
data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.3,random_state=10)
#print(len(data))
#print(len(data_train))
#print(len(data_test))
#print(len(target))
#print(len(target_train))
#print(len(target_test))

gnb = GaussianNB()
#pred = gnb.fit(data_train,target_train).predict(data_test)
#print("Naive-Bayes accuracy:",accuracy_score(target_test,pred,normalize=True))
#
svc_model=LinearSVC(random_state=0)
#pred = svc_model.fit(data_train,target_train).predict(data_test)
#print("LinearSVC accuracy:",accuracy_score(target_test,pred,normalize=True))

neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(data_train,target_train)
#pred=neigh.predict(data_test)
#print("KNeighbors accuracy score:",accuracy_score(target_test,pred,normalize=True))

visualizer = ClassificationReport(gnb,classes=['WON','LOSS'])
visualizer.fit(data_train,target_train)
visualizer.score(data_test,target_test)
g = visualizer.poof()

visualizer = ClassificationReport(svc_model,classes=['WON','LOSS'])
visualizer.fit(data_train,target_train)
visualizer.score(data_test,target_test)
g = visualizer.poof()

visualizer = ClassificationReport(neigh,classes=['WON','LOSS'])
visualizer.fit(data_train,target_train)
visualizer.score(data_test,target_test)
g = visualizer.poof()
