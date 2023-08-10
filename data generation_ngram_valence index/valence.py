# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 21:56:48 2021

@author: ldr
"""


import csv
import pandas as pd
import matplotlib.pyplot as plt

##准备：真正的、从google下载下来的raw data的处理
fa=open(r"C:\Users\ldr\Desktop\googlebooks-eng-all-1gram-20120701-a.tsv",encoding='utf-8')
read_tsva = csv.reader(fa, delimiter="\t")
lista=[]
t=0
for row in read_tsva:
    lista.append(row)
    t=t+1
    if t>=318115:
        break


   
dfa=pd.DataFrame(lista,columns=['1gram','年份','出现次数','包含的书数'])
    
for i in range(len(dfa['1gram'].values)):
    dfa['1gram'].values[i]=dfa['1gram'].values[i].lower()

i=0
for obj in dfa.iloc[:,0]:
    head, sep, tail = obj.partition('_')
    dfa.iloc[i,0]=head
    i=i+1
outputpatha=r"C:\Users\ldr\Desktop\CSS project\case\jpal\valence index\a.csv"
dfa.to_csv(outputpatha,sep=',',index=False,header=True)
#差不多30w数据

##正式数据处理
#将幸福评分表和单词统计表结合
word=pd.read_csv(r"C:\Users\ldr\Desktop\word valence.csv")
word['country'].describe() #unique=5，只保留GB

word=word[word['country']=='GB']
word=word.loc[:,['1gram','valence']]


result = pd.merge(dfa,word,how='outer',on='1gram')
result=result.fillna(0)
result=result[result.loc[:,'年份']!=0]

#计算词频
lines=''
i=0
with open(r'C:\Users\ldr\Desktop\googlebooks-eng-all-totalcounts-20120701.txt', "r") as f:
    for line in f.readlines():
        lines = lines+line.strip('\n') #将导入的txt文件中的数据转化为string


t0=lines.split() #以空白符作为分隔，将各年的数据分隔开
tt=[t.split(',') for t in t0] #以','作为分隔，将每年数据中的各变量（年份、单词总数等）数据分隔开来
fr0=pd.DataFrame(tt,columns=['年份', '总数', 'a','b']) #后两个变量不重要，随便命名，注意与单词统计表中的列名保持一致（“年份”），便于后续融合
fr=fr0.iloc[:,0:2] #只取前两个变量：年份、总数

result1=pd.merge(result,fr,how='outer',on='年份') #融合两个表格，‘outer’意味着取并集，不舍弃数据

result1['出现次数']=pd.to_numeric(result1['出现次数'])
result1['总数']=pd.to_numeric(result1['总数'])
result1['年份']=pd.to_numeric(result1['年份'])


result1['freq']=result1.apply(lambda x:x['出现次数']/x['总数'],axis=1)#计算词频





result1['f_v']=result1.apply(lambda x:x['freq']*x['valence'],axis=1)
r=result1

q=set(r['年份'])

old_list = [2, 3, 4, 5, 1, 2, 3]
new_list = list(set(old_list))
print(new_list) # 不保证顺序：[1, 2, 3, 4, 5]
s=[]
for year in q:
    s.append([year,r.loc[(r['年份']==year),'f_v'].sum()])
    
outcomes=pd.DataFrame(s,columns=['年份','幸福指数'])
x=outcomes['年份']
y=outcomes['幸福指数']
plt.plot(x,y)

