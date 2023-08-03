 # -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:21:12 2021

@author: ldr
"""
##环境准备
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

## 一、导入数据
import numpy as np #往后看到np就是指numpy
import pandas as pd #往后看到pd就是指pandas


df=pd.read_csv(r"C:\Users\ldr\Desktop\CSS project\case\kaggle_churn\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df
##二、描述性分析
df.shape #显示数据的格式
df.dtypes #输出每一列对应的数据类型
df.isnull().sum().values.sum() #查找缺失值
df.nunique() #查看每一列有几个不同值
##三、数据处理
# 将InternetService中的DSL数字网络，fiber optic光纤网络替换为Yes
# 将MultipleLines中的No phoneservice替换成No
replace_list=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for i in replace_list:
    df[i]=df[i].str.replace('No internet service','No')
df['InternetService']=df['InternetService'].str.replace('Fiber optic','Yes')
df['InternetService']=df['InternetService'].str.replace('DSL','Yes')
df['MultipleLines']=df['MultipleLines'].str.replace('No phone service','No')
# 将TotalCharges转换为数字型
df.TotalCharges=pd.to_numeric(df.TotalCharges,errors="coerce") #.to_numeric()将参数转换为数字类型,其中coerce表示无效的解析将设置为NaN
df.TotalCharges.dtypes
##四、数据可视化
#略去 ctrl+1
# 根据(二)中的可视化结果，有11个特征与客户流失率的高低相关，分别是

# SeniorCitizen ：是否老年用户
# Partner ：是否伴侣用户
# Dependents ：是否亲属用户
# tenure： 在网时长
# InternetService：是否开通互联网服务
# OnlineSecurity：是否开通网络安全服务
# TechSupport：是否开通了技术支持服务
# Contract：签订合同方式 （按月，一年，两年）
# PaperlessBilling：是否开通电子账单（Yes or No）
# PaymentMethod：付款方式（bank transfer，credit card，electronic check，mailed check）
# MonthlyCharges：月费用

##五、特征工程
# 第一类特征的数据内容为：‘yes’ or ‘no‘
# 目前属于这类特征的变量有：‘Partner’, ‘Dependents’,‘InternetService’,‘OnlineSecurity’, ‘TechSupport’，‘PaperlessBilling’, ‘Churn’。可以直接采用0-1变量进行编码。其中’1‘代表’yes‘，’0‘代表’no‘

Te_data=df
#将'Partner', 'Dependents','InternetService','OnlineSecurity', 'TechSupport'，'PaperlessBilling', 'Churn'转化为0-1编码
SeniorCitizen=list(Te_data['SeniorCitizen'])
Partner=list(Te_data['Partner'])
Dependents=list(Te_data['Dependents'])
InternetService=list(Te_data['InternetService'])
OnlineSecurity=list(Te_data['OnlineSecurity'])
TechSupport=list(Te_data['TechSupport'])
PaperlessBilling=list(Te_data['PaperlessBilling'])
Churn=list(Te_data['Churn'])


for i in range(Te_data.shape[0]): #df.shape[0] 表示行数
    
    if Partner[i]=='Yes':
        Partner[i] = 1
    else :
        Partner[i] = 0

    if Dependents[i]=='Yes':
        Dependents[i] = 1
    else :
        Dependents[i] = 0

    if InternetService[i]=='Yes':
        InternetService[i] = 1
    else :
        InternetService[i] = 0

    if OnlineSecurity[i]=='Yes':
        OnlineSecurity[i] = 1
    else :
        OnlineSecurity[i] = 0
        
    if TechSupport[i]=='Yes':
        TechSupport[i] = 1
    else :
        TechSupport[i] = 0
        
    if PaperlessBilling[i]=='Yes':
        PaperlessBilling[i] = 1
    else :
        PaperlessBilling[i] = 0
    
    if Churn[i]=='Yes': #流失客户为1
        Churn[i] = 1
    else :
        Churn[i] = 0

# 标称型数据只提供了足够信息区分对象，而本身不具有任何顺序或数值计算的意义。目前属于这类特征的变量有：‘Contract’、‘PaymentMethod’。这类变量采用One-Hot的方式进行编码，构造虚拟变量。
Contract=Te_data['Contract']
Contract_dummies=pd.get_dummies(Contract)#构造变量，从df中抽取出来
PaymentMethod=Te_data['PaymentMethod']
PaymentMethod_dummies=pd.get_dummies(PaymentMethod)
PaymentMethod_dummies


# 数值型数据具备顺序以及加减运算的意义，目前属于这类特征的变量有：tenure，MonthlyCharges。
# 可以采用连续特征离散化的处理方式，因为离散化后的特征对异常数据有更强的鲁棒性，可以降低过拟合的风险，使模型更稳定，预测的效果也会更好。
# 数据离散化也称为分箱操作，其方法分为有监督分箱（卡方分箱、最小熵法分箱）和无监督分箱（等频分箱、等距分箱）。这里采用无监督分箱中的等频分箱进行操作。
tenure=list(Te_data['tenure'])
tenure_cats=pd.qcut(tenure,6) #等频划分，1/6分位数...
tenure_dummies=pd.get_dummies(tenure_cats)
MonthlyCharges=list(Te_data['MonthlyCharges'])
MonthlyCharges_cats=pd.qcut(MonthlyCharges,5)
MonthlyCharges_dummies=pd.get_dummies(MonthlyCharges_cats)
tenure_dummies

##六、得到输入输出特征
import numpy as np #把list变成array
#模型输出y
Churn_y=np.array(Churn).reshape(-1,1) #.reshape转换成1列，-1为模糊值，行数=总维数÷列数

#模型输入x：'SeniorCitizen', 'Partner', 'Dependents','InternetService','OnlineSecurity', 'TechSupport'，'PaperlessBilling','Contract','PaymentMethod','tenure',MonthlyCharges
SeniorCitizen_x=np.array(SeniorCitizen).reshape(-1,1)
Partner_x=np.array(Partner).reshape(-1,1)
Dependents_x=np.array(Dependents).reshape(-1,1)
InternetService_x=np.array(InternetService).reshape(-1,1)
OnlineSecurity_x=np.array(OnlineSecurity).reshape(-1,1)
TechSupport_x=np.array(TechSupport).reshape(-1,1)
PaperlessBilling_x=np.array(PaperlessBilling).reshape(-1,1)

Contract_x=Contract_dummies.values
PaymentMethod_x=PaymentMethod_dummies.values

tenure_x=tenure_dummies.values
MonthlyCharges_x=MonthlyCharges_dummies.values

X=np.concatenate([SeniorCitizen_x,Partner_x,Dependents_x,InternetService_x,OnlineSecurity_x,TechSupport_x,TechSupport_x,Contract_x,PaymentMethod_x,tenure_x,MonthlyCharges_x],axis=1)

##！七、训练与测试
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Churn_y, test_size=0.3, random_state=40)
 #以30%和70%的比例，分别将X和Churn_y（标签）划分成训练集和测试集
#random_state:随机种子，限定为40（可以是任意数），是为了使得重复运行时结果不会改变

# 构建模型
tree = DecisionTreeClassifier(max_depth=6,random_state=0) #树的深度设置为6
dt_tree=tree.fit(x_train,y_train)  
#tree.fit(X,y)：监督学习，用X和y来训练模型，返回值dt_tree就是我们需要的决策树模型
dt_tree
#评估模型使用十次交叉验证
score = cross_val_score(tree, X, Churn_y, cv=10, scoring='accuracy')

print("training set score:{:.3f}".format(tree.score(x_train,y_train)))
print("test set score:{:.3f}".format(tree.score(x_test,y_test)))
print("ten cross-validation score:{:.3f}".format(np.mean(score)))
print("Feature importances : \n{}".format(tree.feature_importances_)) ##系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大


##结果可视化
# 由于X是array格式的，没有列名，首先将X转换为dateframe格式，再加上列名，并将其保存到xlsx表中。

# 首先将array转换为framedata
Te_X = pd.DataFrame(X)
# 加上列名
Te_X.columns=['SeniorCitizen', 'Partner', 'Dependents','InternetService','OnlineSecurity', 'TechSupport','PaperlessBilling',\
       'Month-to-month','One year','Two year','Bank transfer (automatic)','Credit card (automatic)','Electronic check','Mailed check',\
        'tenure_(0.999, 4.0]','tenure_(4.0, 14.0]','tenure_(14.0, 29.0]','tenure_(29.0, 47.0]','tenure_(47.0, 64.0]','tenure_(64.0, 72.0]',\
      'MonthlyCharges_(18.249, 25.05]','MonthlyCharges_(25.05, 58.92]','MonthlyCharges_(58.92, 79.15]','MonthlyCharges_(79.15, 94.3]','MonthlyCharges_(94.3, 118.75]']
Te_X
# 同理对Churn_y
Churn_y=pd.DataFrame(Churn_y,columns=['Churn'])
Churn_y

# 保存dataframe数据到xlsx文件
Te_X.to_excel("tree_data.xlsx", index=0)
Churn_y.to_excel("tree_Y.xlsx", index=0)

#特征重要性可视化  
import matplotlib.pyplot as plt #matplotlib：python中的画图库
def plot_feature_importance(model): #定义一个可以展示特征重要性的绘图函数
    n_features = Te_X.shape[1]  
    plt.barh(range(n_features),model.feature_importances_,align='center')#绘制条形图，条形高度这个参数为模型的特征重要性
    plt.yticks(range(n_features),Te_X.columns[0:]) #定义y轴刻度值，x轴特殊未定义，显示数字刻度
    plt.xlabel('Features importance') #x轴标签
    plt.ylabel('feature') #y轴标签
plot_feature_importance(tree)
plt.show() #展示图片

#决策树可视化
from sklearn.tree import export_graphviz
export_graphviz(tree,out_file='te_tree.dot',class_names=['Churn_yes','Churn_no'],feature_names=Te_X.columns[0:],impurity=False,filled=True)
import graphviz #需要安装Graphviz才能使用
#Graphviz安装教程：https://www.cnblogs.com/linfangnan/p/13210536.html
with open("te_tree.dot") as f:
    dot_graph=f.read()
graph=graphviz.Source(dot_graph)
graph.render("tree") #输出'tree.pdf'




