
# coding: utf-8

# In[3]:


#参考サイト
#http://tekenuko.hatenablog.com/entry/2016/09/20/222453
#https://www.codexa.net/kaggle-titanic-beginner/
#https://www.slideshare.net/hamadakoichi/randomforest-web
#http://www.randpy.tokyo/entry/python_random_forest   ←メイン


# In[4]:


#必要なモジュールをインポート
#サイキットラーンは、from sklearn import datasets　という他とは違う感じらしい


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


# In[ ]:


#from sklearn import datasets
#sklearnが提供しているデータを使うときは↑のコードが必要


# In[6]:


#train.csvというcsvファイルを読み込む
#タイタニック号の乗客データ


# In[7]:


train = pd.read_csv('train.csv')
train


# In[8]:


#「passengerld」乗客のid
#「Survived」生存したか死亡したか（生存：１、死亡：０）
#「Pclass」部屋のランク
#「Name」乗客の名前
#「Sibsp」乗車していた兄弟・配偶者の数
#「Parch」乗車していた親・子どもの数
#「Ticket」チケット番号
#「Fare」運賃
#「Cabin」客室番号
#「Embarked」乗船した港（Cherbourg,Queenstown,Southampton）


# In[9]:


#欠損値を０で補完する


# In[10]:


train = train.fillna(0)
train.head()


# In[11]:


#データフレーム内のstringをfloatに変える
#男性を0、女性を1とする
#Sを0、Cを１、Qを２とする


# In[12]:


train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S" ] = 0
train["Embarked"][train["Embarked"] == "C" ] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

train.head()


# In[13]:


#使わないカラムを削除する
#使わないのは、名前、客室番号、乗客のid、チケット番号の４つ


# In[14]:


train = train.drop(['Name','Cabin','PassengerId','Ticket'],axis=1)
train.head()


# In[15]:


#目的変数と説明変数に分ける


# In[16]:


X = train.drop('Survived',axis=1)
y = train.Survived


# In[17]:


#trainデータとtestデータに分割する


# In[18]:


from sklearn.model_selection import train_test_split
y_train,y_test,X_train,X_test = train_test_split(y,X,test_size=0.4)


# In[19]:


#★★RandamForestClassifier★★


# In[20]:


#ランダムフォレストという学習器を定義する
#レジュメにある「RandomForestRegreddor」ではなく、「RandomForestClassifier」を使う
#trainデータで学習器をfittingする


# In[21]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train,y_train)


# In[22]:


#fitされた学習器にtestデータを当てはめてみる
#X_testを当てはめて予測を行う


# In[23]:


y_pred = clf.predict(X_test)


# In[24]:


#精度を計算する


# In[25]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# In[26]:


#★★LogisticRegression★★


# In[27]:


#LogisticRegressionという学習器を定義する
#trianデータで学習器をfittingする


# In[29]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0, penalty='l2')
clf.fit(X_train,y_train)


# In[30]:


#fitされた学習器にtestデータを当てはめてみる
#X_testを当てはめて予測を行う


# In[31]:


y_pred = clf.predict(X_test)


# In[32]:


#精度を計算する


# In[33]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# In[34]:


#★★SVC★★


# In[35]:


#SVCという学習器を定義する
#trainデータで学習器をfittingする


# In[36]:


from sklearn import svm
clf = svm.SVC()
clf.fit(X_train,y_train)


# In[37]:


#fitされた学習器にtestデータを当てはめてみる
#X_testを当てはめて予測を行う


# In[38]:


y_pred = clf.predict(X_test)


# In[39]:


#精度を計算する


# In[40]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

