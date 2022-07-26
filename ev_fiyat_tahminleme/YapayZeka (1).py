#!/usr/bin/env python
# coding: utf-8

# # YBS4015 YAPAY ZEKA PROJESİ

# ## Lineer Regresyon İle Ev Fiyatı Tahminleme Çalışması  

# #### 1.Kütüphanelerin import edilmesi 

# In[33]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# #### 2.Verisetinin import edilmesi

# In[3]:


from sklearn import datasets
boston= datasets.load_boston()


# #### 3.Verisetine genel bakış

# In[4]:


#Satır,Sütun Sayısı
print(boston.data.shape)


# In[6]:


#Kolon isimleri
print(boston.feature_names)


# In[7]:


#Tanımlama
print(boston.DESCR)


# In[56]:


#Verisetinin DataFrame formatına dönüştürülmesi ve Fiyat kolonunun eklenmesi
data= pd.DataFrame(boston.data, columns= boston.feature_names)
data["Price"]= boston.target
data.head(10)


# In[57]:


#Eksik değerlerin belirlenmesi
data.isnull().sum()


# In[58]:


#Tanımsal İstatistikler
data.describe().T


# In[59]:


#Veri Dağılımlarının Görselleştirilmesi
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(15,10))
plt.show()


# #### 4. 1.Modelin kurulması 

# In[60]:


#Verinin Eğitim ve Test verisi olarak ikiye bölünmesi
x= boston.data
y= boston.target

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=0.2, random_state=0)

print("xtrain shape : ", xtrain.shape)
print("xtest shape : ", xtest.shape)
print("ytrain shape : ", ytrain.shape)
print("ytest shape : ", ytest.shape)


# In[61]:


#Modelin Çalıştırılması
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(xtrain, ytrain)

y_pred= regressor.predict(xtest)


# In[62]:


#Sonuçların Görselleştirilmesi
plt.scatter(ytest, y_pred)
plt.xlabel("1000$ Ev fiyatları")
plt.ylabel("Tahminlenen Değerler")
plt.title("Gerçek Değer vs Tahminlenen Değer")
plt.show()


# #### 5. 1.Modelin Performansı

# In[63]:


#Modelin Performansı
rmse2 = (np.sqrt(mean_squared_error(ytest, y_pred)))                      


print("Eğitim Seti için Modelin Performansı")
print("Fiyat Tahmini")
print("--------------------------------------")
print('Hata Kareleri Ortalaması(RMSE): {}'.format(rmse2))
print("--------------------------------------")
reKare= round(r2_score(ytest, y_pred),2)
print("R2 Skoru: ",reKare)


# In[64]:


#


# ### 6. 2. Model

# In[65]:


#Isı Haritası 
bos_1= pd.DataFrame(boston.data, columns= boston.feature_names)

correlation_matrix= bos_1.corr().round(2)
plt.figure(figsize=(13,10))
sns.heatmap(data=correlation_matrix, annot=True)


# In[74]:


#LSTAT ve RM Değişkenlerinin Görselleştirilmesi
print("(LSTAT= Düşük Gelir Düzeyi / RM= Oda Sayısı)")
plt.figure(figsize=(20,5))
features= ['LSTAT', 'RM']
target= data['Price']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x= data[col]
    y= target
    plt.scatter(x,y, marker='o')
    plt.title("Ev Fiyatlarının Dağılımı")
    plt.xlabel(col)
    plt.ylabel("Ev Fiyatları(1000$)")


# #### 7. 2.Modelin Kurulması

# In[79]:


#Fiyat ve Oda Sayısı Değişkenlerinin Array formuna aktarılması
X_rooms= data.RM
y_price= data.Price

X_rooms= np.array(X_rooms).reshape(-1,1)
y_price= np.array(y_price).reshape(-1,1)

print(X_rooms.shape)
print(y_price.shape)


# In[83]:


#Versetinin Eğitim Ve Test Olarak ikiye Bölünmesi
X_train_1, X_test_1, Y_train_1, Y_test_1= train_test_split(X_rooms, y_price, test_size=0.2, random_state=5)
print(X_train_1.shape)
print(X_test_1.shape)
print(Y_train_1.shape)
print(Y_test_1.shape)


# In[84]:


#Modelin Çalıştırılması
reg_1 = LinearRegression()
reg_1.fit(X_train_1, Y_train_1)

y_train_predict_1 = reg_1.predict(X_train_1)


# In[86]:


#Modelin Görselleştirilmesi
from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=80)
prediction_space= np.linspace(min(X_rooms),
max(X_rooms)).reshape(-1,1)
plt.scatter(X_rooms, y_price)
plt.plot(prediction_space, reg_1.predict(prediction_space),
color= 'red', linewidth=3)
plt.ylabel("Ev Fiyatları (1000$)")
plt.xlabel("Oda Sayısı")
plt.show()


# #### 8. 2.Modelin Performansı

# In[89]:


#RMSE ve R2 Ölçütleri
rmse = (np.sqrt(mean_squared_error(Y_train_1, y_train_predict_1)))
r2 = round(reg_1.score(X_train_1, Y_train_1),2)

print("2.Modelin Performansı")
print("--------------------------------------")
print('Hata Kareleri Ortalaması(RMSE): {}'.format(rmse))
print("--------------------------------------")
print('R2 Skoru: {}'.format(r2))
print("\n")


# #

# Anıl Alkan - 2018469005
# Sefa Başıbütün - 2018469016
# Hasan Can Çelik - 2018469068

# 

# In[ ]:




