#!/usr/bin/env python
# coding: utf-8

# ### Fake News Classifier
# Dataset:  https://www.kaggle.com/c/fake-news/data#

# In[2]:


import pandas as pd


# In[5]:


df=pd.read_csv(r'C:\Users\Maheswari\Downloads\mm\train.csv')


# In[6]:


df.head()


# In[7]:


## Get the Independent Features

X=df.drop('label',axis=1)


# In[8]:


X.head()


# In[9]:


## Get the Dependent features
y=df['label']


# In[10]:


y.head()


# In[11]:


df.shape


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


# In[13]:


df=df.dropna()


# In[14]:


df.head(10)


# In[15]:


messages=df.copy()


# In[16]:


messages.reset_index(inplace=True)


# In[17]:


messages.head(10)


# In[18]:


messages['text'][6]


# In[ ]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[ ]:


corpus[3]


# In[ ]:





# In[ ]:


## TFidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=tfidf_v.fit_transform(corpus).toarray()


# In[18]:


X.shape


# In[19]:


y=messages['label']


# In[20]:


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[21]:


tfidf_v.get_feature_names()[:20]


# In[22]:


tfidf_v.get_params()


# In[23]:


count_df = pd.DataFrame(X_train, columns=tfidf_v.get_feature_names())


# In[27]:


count_df.head()


# In[28]:


import matplotlib.pyplot as plt


# In[29]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### MultinomialNB Algorithm

# In[30]:



from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[31]:


from sklearn import metrics
import numpy as np
import itertools


# In[32]:



classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[33]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score


# In[34]:


y_train.shape


# ### Passive Aggressive Classifier Algorithm

# In[35]:


from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(n_iter=50)


# In[ ]:


linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])


# ### Multinomial Classifier with Hyperparameter

# In[37]:


classifier=MultinomialNB(alpha=0.1)


# In[ ]:


previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))


# In[106]:


## Get Features names
feature_names = cv.get_feature_names()


# In[109]:


classifier.coef_[0]


# In[107]:


### Most real
sorted(zip(classifier.coef_[0], feature_names), reverse=True)[:20]


# In[135]:


### Most fake
sorted(zip(classifier.coef_[0], feature_names))[:5000]


# ## HashingVectorizer
# 

# In[ ]:


hs_vectorizer=HashingVectorizer(n_features=5000,non_negative=True)
X=hs_vectorizer.fit_transform(corpus).toarray()


# In[43]:


X.shape


# In[47]:


X


# In[44]:


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[45]:



from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[ ]:




