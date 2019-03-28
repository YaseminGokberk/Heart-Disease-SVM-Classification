# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:56:35 2019

@author: yasem
"""

"""
 HEART DISEASE - SVM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
import itertools
import seaborn
from sklearn.metrics import r2_score

## Confusion Matrix grafigi icin fonk
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


data = pd.read_csv('heart_disease.csv')
data.head()
data.target.value_counts()

seaborn.countplot(x="target", data=data, palette="bwr")

class_names = np.array(['0', '1'])


# VERI ON ISLEME

#x = data.iloc[:,0:-1]
x = data.drop(labels="target", axis = 1)
y= data.iloc[:,-1:].values


## OZNITELIK OLCEKLEME ##
# NORMALIZE ISLEMI
x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)).values

## OZNITELIK OLCEKLEME ##

#VERILERIN EGITIM-TEST OLARAK BOLUNMESI
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size= 0.33, random_state=0 )
## x: bagımsız degisken, y: bagımlı degisken, test verisi oranı: test_size, bolunme sekli: random_state

"""
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) #tekrar egitme sadece transform et
## standartlastırma islemi yapıldı
## basarılı olmadıgı icin normalize islemi tercih edildi
"""

# Target 'taki 0 ve 1 degerlerinin sayıları

no_heart_disease = y[y[:,0]== 0]
no_heart_disease = len(no_heart_disease)
heart_disease = y[y[:,0]==1]
heart_disease = len(heart_disease)
print("Sıfırlar: "+str(no_heart_disease)+"\nBirler: "+str(heart_disease))


# MODEL -> SVM CLASSIFICATION (kernel : linear)
classification_ln = SVC(kernel="linear") 
classification_ln.fit(x_train, y_train)
predicts_ln = classification_ln.predict(x_test)


#CONFUSION MATRIX - LINEAR SVM

cm_ln = confusion_matrix(y_test, predicts_ln)
print("Confusion Matrix (linear kernel):\n"+str(cm_ln))
plot_confusion_matrix(cm_ln, class_names)

#print("Accuracy: "+str( (cm_ln[0][0]+cm_ln[1][1]) / (sum(cm_ln[0])+sum(cm_ln[1])) ) )
print("Test Accuracy of SVM Algorithm (Linear): {:.2f}%".format(classification_ln.score(x_test,y_test)*100))

# MODEL -> SVM CLASSIFICATION (kernel : rbf)
classification_rbf = SVC(kernel="rbf", )
classification_rbf.fit(x_train, y_train)
predicts_rbf = classification_rbf.predict(x_test)

#import statsmodels.formula.api as sm 
#r_ols = sm.OLS(endog=y, exog=x)
#r= r_ols.fit()
#print(r)

#CONFUSION MATRIX - RBF SVM

cm_rbf = confusion_matrix(y_test, predicts_rbf)
print("Confusion Matrix (RBF Kernel):\n"+str(cm_rbf))
plot_confusion_matrix(cm_rbf, class_names)
#print("Accuracy: "+str( (cm_rbf[0][0]+cm_rbf[1][1]) / (sum(cm_rbf[0])+sum(cm_rbf[1])) ) )
print("Test Accuracy of SVM Algorithm (RBF): {:.2f}%".format(classification_rbf.score(x_test,y_test)*100))

# R2 SCORE
linear_r2 = r2_score(y_test, predicts_ln)
rbf_r2 = r2_score(y_test, predicts_rbf)

print("Linear SVM R2 degeri:"+str(linear_r2))
print("Linear SVM R2 degeri:"+str(rbf_r2))

