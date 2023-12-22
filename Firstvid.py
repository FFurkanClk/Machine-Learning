# -*- coding: utf-8 -*-
#Kütüphaneler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Kodlar

#Veri Yükleme

# veriler = pd.read_csv('veriler.csv')
#veriler = pd.read_csv('veriler.csv')
#print(veriler)

#Veri Ön İşleme
#boy = veriler['boy']
#print(boy)
#x = 10
#
#boykilo = veriler[['boy','kilo']]
#print(boykilo)

#class insan:
   # boy = 180
    #def kosmak(self,b):
        #return b+10
#ali = insan()
#print(ali.boy)
#print(ali.kosmak(90))    
#l = [1,2,3,4,5]

#eksik veriler
eksikveriler = pd.read_csv('eksikveriler.csv')
print(eksikveriler)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
yas = eksikveriler.iloc[:,1:4].values
print(yas)
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print(yas)

ulke = eksikveriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] =le.fit_transform(eksikveriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

print(list(range(22)))
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'])
print(sonuc)

sonuc2= pd.DataFrame(data=yas, index = range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = eksikveriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index= range(22), columns= ['cinsiyet'])
print(sonuc3)

s = pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
