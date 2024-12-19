# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Dataset/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Preprocessing the data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split() # Splitting the review into a list of words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review) # Joining the list of words 
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
NaiveBayes = GaussianNB()
NaiveBayes.fit(X_train, y_train)

# Predicting the Test set results of the Naive Bayes Classification
y_predNaiveBayes = NaiveBayes.predict(X_test)

# Making the Confusion Matrix for the Naive Bayes Classification
from sklearn.metrics import confusion_matrix
cmNB = confusion_matrix(y_test, y_predNaiveBayes)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
RandomForest.fit(X_train, y_train)

# Predicting the Test set results of the Random Forest Classification
y_predRandomForest = RandomForest.predict(X_test)

#Making the Confusion Matrix for the Random Forest Classification
cmRF = confusion_matrix(y_test, y_predRandomForest)

# Applying k-Fold Cross Validation for Naive Bayes Classification
from sklearn.model_selection import cross_val_score
accuracies_naivebayes = cross_val_score(estimator = NaiveBayes, X = X_train, y = y_train, cv = 10)
accuracies_naivebayes.mean()
accuracies_naivebayes.std()

# Applying k-Fold Cross Validation for Random Forest Classification
from sklearn.model_selection import cross_val_score
accuracies_randomforest = cross_val_score(estimator = RandomForest, X = X_train, y = y_train, cv = 10)
accuracies_randomforest.mean()
accuracies_randomforest.std()
