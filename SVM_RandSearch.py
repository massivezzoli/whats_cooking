
import json
import pandas as pd
import numpy as np
from scipy.stats import randint as randint
import nltk
import re
import itertools
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import random
import matplotlib.pyplot as plt
from utils import*

#Load Data
traindf = pd.read_json("train.json")
testdf = pd.read_json("test.json")

cuis_ingr = {}
ingr_list=[]
for a,b in traindf.groupby('cuisine'):
    cuis_ingr[a] = list(itertools.chain.from_iterable(b['ingredients'].values))
    ingr_list+=list(itertools.chain.from_iterable(b['ingredients'].values))

#### create list of ingredients to remove:
#list ingredients that appear only one time in the dataset
unique_ingr = pd.Series(ingr_list).value_counts()
ingr_rm= unique_ingr[unique_ingr<2].index.tolist()

#### Clean train dataset:
traindf['ingredients_rm'] = traindf.apply(lambda row: remove_ing(row['ingredients'],ingr_rm), axis=1)
traindf['ingredients_string'] = traindf.apply(lambda row: preprocess_ing(row['ingredients_rm']), axis=1)
#### Clean test dataset:
testdf['ingredients_rm'] = testdf.apply(lambda row: remove_ing(row['ingredients'],ingr_rm), axis=1)
testdf['ingredients_string'] = testdf.apply(lambda row: preprocess_ing(row['ingredients_rm']), axis=1)

#### Create Corpus and Vecotrize ingredients
corpus_train = traindf['ingredients_string']
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1),
                             analyzer='word', max_df=0.5, token_pattern=r'\w+')

train_vector = vectorizer.fit_transform(corpus_train)
train_vector_feat = vectorizer.get_feature_names()
train_labels = traindf['cuisine']
corpus_test = testdf['ingredients_string']
test_vector = vectorizer.transform(corpus_test)

# ### Create Train-Validation Split
X_train, X_val, y_train, y_val=tts(train_vector, train_labels,
                                   test_size=0.2, random_state=42)

#### SVC
#define parameters to be searched
param_dist = {'C': np.linspace(0.1, 20),
              'gamma':   np.linspace(0.1, 20),
              'kernel': ['rbf','linear']}

# call classifier
svc=SVC(class_weight= 'balanced', decision_function_shape='ovr', random_state=42)

rs = RandomizedSearchCV(estimator=svc, param_distributions=param_dist,
                        n_iter=1, verbose=1, n_jobs=-1)

print('Searching parameters...')
rs.fit(X_train, y_train)
print('Best Score:',rs.best_score_)
print('Best parameters:',rs.best_params_)

#### Fit best model and report val score:
svc_model = rs.best_estimator_
svc_model.fit(X_train, y_train)
print('Validation accuracy: %.3f' % svc_model.score(X_val, y_val))

input('press enter to make test prediction')
#### Predict Test and save results
prediction_svc = svc_model.predict(test_vector)
sub_svc = testdf[['id']].copy()
sub_svc['cuisine']= prediction_svc
sub_svc.to_csv("submission_svc_rs.csv",index=False)

#### Plot confusion Matrix:
input('press enter to save confusion matrix')

cm = confusion_matrix(y_val, svc_model.predict(X_val))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 10))
cuisines = traindf['cuisine'].value_counts().index.tolist()
xt=yt=sorted(cuisines)

plt.figure(figsize=(12,10))
with sns.axes_style("white"):
    ax = sns.heatmap(cm_normalized, square=True,xticklabels=xt, yticklabels=yt,
                     cmap='YlGnBu', annot=True, fmt='.2f',linewidths=.5)
ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')
ax.figure.savefig("optimal_svc_cm.png")
