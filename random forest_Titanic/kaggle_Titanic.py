# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:12:14 2021

@author: ldr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #作图包
import seaborn as sns #在matplotlib基础上进行了更高级的API封装,使得作图更加容易
import warnings #这干啥的？
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']#用来显示正常中文标签

train=pd.read_csv(r'C:\Users\ldr\Desktop\CSS project\case\kaggle_Titanic\train.csv')
test=pd.read_csv(r'C:\Users\ldr\Desktop\CSS project\case\kaggle_Titanic\test.csv')

train.head()
train.info()
train.isna().sum() #统计null值个数
train.isnull().sum()

train["Survived"].value_counts()# 统计：549丧生 342存活
##数据可视化
#性别-生存概率
pd.pivot_table(train, index=["Sex"],values=["Survived"]).plot.bar()
sns.barplot(x='Sex', y='Survived', data=train) #这个好看多了
pd.crosstab(train.Sex,train.Survived).plot.bar(stacked=True)
#年龄-生存概率
#pd.crosstab(train.Age,train.Survived).plot.bar(stacked=True) 太密了，可以设置一下间隙
age_sur=sns.FacetGrid(train,hue='Survived',aspect=2.5)
age_sur.map(sns.kdeplot,'Age',shade=True)
age_sur.set(xlim=(0,train['Age'].max()))
age_sur.add_legend()
#舱等级-生存概率
sns.barplot(x='Pclass',y='Survived',data=train)
pd.crosstab(train.Pclass,train.Survived).plot.bar(stacked=True)

train_m=train[train.Sex=='male']#男性
pd.crosstab(train_m.Pclass,train_m.Survived).plot.bar(stacked=True)
sns.barplot(x='Pclass',y='Survived',data=train_m)
train_f=train[train.Sex=='female']#女性
pd.crosstab(train_f.Pclass,train_f.Survived).plot.bar(stacked=True)
sns.barplot(x='Pclass',y='Survived',data=train_f)


#登船地点-生存概率
sns.barplot(x='Embarked',y='Survived',data=train)
sns.countplot('Embarked',hue='Survived',data=train) #每个港口的生存丧生人数比较
#(趣味附加）各舱位年龄分布
train.Age[train.Pclass==1].plot(kind='kde')
train.Age[train.Pclass==2].plot(kind='kde')
train.Age[train.Pclass==3].plot(kind='kde')

plt.title(u"不同等级舱位上的年龄分布")

##填充缺失值
train['Embarked'].value_counts()
train['Embarked']=train['Embarked'].fillna('S')
#pd.pivot_table(train,index='Cabin',values='Survived').plot.bar()


#随机森林法填充年龄
from sklearn.ensemble import RandomForestRegressor #ensemble learning：集成学习

def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df['Age'].notnull()].values #as.matrix()已经没了，用.values替代，注意，没有（），因为是属性
    unknown_age = age_df[age_df['Age'].isnull()].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)#n_jobs：有多少处理器可以使用，-1意味着没有限制
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

# def f(df):
#     df1=df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
#     train=df1[df['Age'].notnull()].as_matrix
#     test=df1[df['Age'].isnull()].as_matrix
    
#     X=train[:,1:]
#     y=train[:,1]
    
#     rfr=RandomForestRegression(random_state=0,n_estimators=2000,n_jobs=-1)
#     rfr.fit(X,y)
    
#     y0=rfr.predict(test[:,1:])
#     df.loc[(df['Age'].isnull()),'Age']=y0
    
#     return df,rfr

train, rfr = set_missing_ages(train) #train, rfr就是函数返回的两个值（补全的数据集和模型）

#将cabin的nan转化为no，舱位号统一为yes
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
train = set_Cabin_type(train)

##特征工程
#选择变量
cor=train.corr()  #各变量相关系数矩阵
print(cor['Survived'].sort_values(ascending=False)) #只看Survived那一列
#passengerID = -0.005007，和Survived无关

#one-hot编码
d_Pclass=pd.get_dummies(train['Pclass'],prefix='Pclass')
d_Sex=pd.get_dummies(train['Sex'],prefix='Sex')
d_Cabin=pd.get_dummies(train['Cabin'],prefix='Cabin')
d_Embarked = pd.get_dummies(train['Embarked'], prefix= 'Embarked')

#构建数据集
df=pd.concat([train,d_Pclass,d_Sex,d_Cabin,d_Embarked],axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis=1,inplace=True)

Xt=df.loc[:,'Age':]
yt=df.loc[:,'Survived']


#预处理test数据集
test.isnull().sum()
test.loc[ (test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[test.Age.isnull()].values
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
test.loc[ (test.Age.isnull()), 'Age' ] = predictedAges

test = set_Cabin_type(test)
dummies_Cabin = pd.get_dummies(test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
X_test=df_test.loc[:,'Age':]


##建模1 logistic
# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
from sklearn import linear_model
# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(tol=1e-6)
clf.fit(Xt, yt)
    
clf
#预测1
predictions_lo = clf.predict(X_test)
result_lo = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':predictions_lo.astype(np.int32)})

#建模2 KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)      # 创建一个KNN的模型
neigh.fit(Xt, yt)    
#预测2
predictions_knn = clf.predict(X_test)

result_knn = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':predictions_knn.astype(np.int32)})

#建模3 SVM
from sklearn import svm
 clf_svm = svm.SVC(kernel='rbf', gamma = 'scale')
clf_svm.fit(Xt,yt) 
predictions_svm=clf_svm.predict(X_test)
 result_svm = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':predictions_svm.astype(np.int32)})
 
