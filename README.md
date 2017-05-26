# whats_cooking
Small project on Kaggle "What's Cooking" dataset

The preliminary data exploration work is contained in the file Data Exploration.ipynb. Preprocessing_Classification.ipynb containes the data cleaning routines, the fitting of the train data with the algorithms selected 
and the predictions of labels for the test set. Logistic and NB classifiers have an example of the RandomizedSearch performed. 
At the bottom of the notebook  a confusion matrix can be generated.
SVM_RandSearch.py  contains the routine used for the hyperparameter search.

The data sets downloaded from Kaggle are test.json and train.json.

For this work Python 3.6 was used. Also, the main packages used are:
numpy 1.12.0,
pandas 0.19.2,
sklearn 0.18.1,
nltk 3.2.2,
seaborn 0.7.1,
matplotlib 2.0,
