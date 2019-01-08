#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Read data
import pandas as pd
X_train = pd.read_csv('train-data.csv')

X_test = pd.read_csv('test-data.csv')

Y_train = pd.read_csv('train-targets.csv')

Y_test = pd.read_csv('test-targets.csv')
len(X_train), len(X_test)


# In[2]:


#Train Model without cross validation
from sklearn.svm import SVC
model = SVC(C=10, kernel='rbf', gamma = 0.02)


# In[3]:


model.fit(X_train, Y_train.values.ravel())

y_pred = model.predict(X_test)

y_pred


# In[4]:


# Report scores and accuracy
from sklearn import metrics

report = metrics.classification_report(Y_test, y_pred)

print(report)


# In[5]:


print("Model Accuracy without cross validation is : ", metrics.accuracy_score(Y_test, y_pred))


# In[13]:


##### Trainin with cross validation

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits = 3, shuffle = True, random_state =365)

gammas = [0.1, 0.05, 0.02, 0.01]
accuracy_scores = []

for gamma in gammas:
    model = SVC(C= 10, kernel = 'rbf', gamma=gamma)
    scores = cross_val_score(model, X_train, Y_train.values.ravel(), cv = kf, scoring = 'accuracy')
    accuracy_score = scores.mean()
    accuracy_scores.append(accuracy_score)
    Pscores = cross_val_score(model, X_train, Y_train.values.ravel(), cv = kf, scoring = 'precision')
    precision_score = Pscores.mean()
    Rscores = cross_val_score(model, X_train, Y_train.values.ravel(), cv = kf, scoring = 'recall')
    recall_score = Rscores.mean()
    Fscores = cross_val_score(model, X_train, Y_train.values.ravel(), cv = kf, scoring = 'f1')
    f1_score = Fscores.mean()
    print("For Gamma : " , gamma , " Accuracy is: " ,accuracy_score)
    print("For Gamma : " , gamma , " Precision is: " ,precision_score)
    print("For Gamma : " , gamma , " Recall is: " ,recall_score)
    print("For Gamma : " , gamma , " F1-score is: " ,f1_score)
    
best_index = np.array(accuracy_scores).argmax()
best_gamma = gammas[best_index]

model = SVC(C=10, kernel = 'rbf', gamma = best_gamma)
model.fit(X_train, Y_train.values.ravel())

y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, y_pred)

print("Best Accuracy is: " ,accuracy)

print("For Best Gamma value: " , best_gamma)


# In[7]:


report = metrics.classification_report(Y_test, y_pred)

print(report)


# In[8]:


df = pd.DataFrame(y_pred)

df.to_csv("test-pred.txt", index = False)


# In[9]:


#Learning Curve

from sklearn.model_selection import learning_curve
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.figure()
plt.title('Learning Curve')
plt.xlabel("Training Examples")
plt.ylabel('Score')
plt.grid()

model = SVC(C=10, kernel='rbf', gamma = best_gamma)

train_sizes, train_scores, val_scores = learning_curve(model, X_train, Y_train.values.ravel(), 
                                                      scoring = 'accuracy', cv = 3)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color = "r", label = "Training Score")

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                 train_scores_mean+train_scores_std, alpha = 0.1, color="r")

plt.plot(train_sizes, val_scores_mean, 'o-', color= 'g', label= 'Cross-Validation Score')

plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                val_scores_mean+val_scores_std, alpha = 0.01, color = 'g')

plt.ylim(0.05, 1.3)
plt.legend()
plt.savefig("SpamLearningCurve.png", dpi= 300)
plt.show()


# In[10]:


# Extra: PCA
import numpy as np
from sklearn.decomposition import PCA
X = X_train
pca = PCA(n_components=2)
pca.fit(X)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
print(pca.explained_variance_ratio_)  

print(pca.singular_values_)  


# In[11]:


# Visualizing training data
plt.figure()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(X_train)
plt.scatter(proj[:, 0], proj[:, 1], c=Y_train.values.ravel(), cmap="Paired")
plt.colorbar()
plt.savefig("SpamVisualize.png", dpi= 300)


# In[12]:


print(sum(Y_train.values.ravel() == 0))
print(sum(Y_train.values.ravel() == 1))







