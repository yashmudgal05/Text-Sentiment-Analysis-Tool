#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git


# In[2]:


# !python -m spacy download en_core_web_lg


# In[68]:


import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


# In[2]:


import preprocess_kgptalkie as kgp


# In[ ]:





# ## working with dataset

# In[3]:


df = pd.read_csv("model.csv", encoding='latin', names = ['polarity','id','date','query','user','text'])
df.sample(5)


# In[4]:


df["polarity"].value_counts()


# In[5]:


df1 = df.loc[1:100000, :]


# In[6]:


df1["polarity"].value_counts()


# In[7]:


df2 = df.loc[800000:899999, :]


# In[8]:


df2["polarity"].value_counts()


# In[9]:


df = df1.append(df2, ignore_index=True)


# In[10]:


df["polarity"].value_counts()


# In[11]:


df.drop('id', inplace=True, axis=1)
df.drop('date', inplace=True, axis=1)
df.drop('query', inplace=True, axis=1)
df.drop('user', inplace=True, axis=1)
df.reset_index(drop=True, inplace=True)


# In[12]:


df.head(5)


# In[14]:


df.to_csv('newModel.csv')


# In[ ]:





# ## working with new data set

# In[15]:


df = pd.read_csv("newModel.csv", encoding='latin', usecols= ["polarity", "text"])


# In[16]:


df.head(10)


# In[17]:


df.tail(10)


# In[18]:


df.shape


# In[19]:


# 0 -> negative
# 4 -> positive
df['polarity'].value_counts()


# In[20]:


def get_clean(x):
    x = str(x).lower().replace('\\', ' ').replace('_', ' ').replace('.', ' ')
    x = kgp.cont_exp(x)
    x = kgp.remove_emails(x)
    x = kgp.remove_urls(x)
    x = kgp.remove_html_tags(x)
    x = kgp.remove_rt(x)
    x = kgp.remove_accented_chars(x)
    x = kgp.remove_special_chars(x)
    x = kgp.remove_dups_char(x)
    x = kgp.make_base(x)
    return x


# In[21]:


tw = get_clean("Retweet you \ _ .will never regrets it Smiling face with heart-shaped eyes")
tw


# In[22]:


get_ipython().run_cell_magic('time', '', "df['text'] = df['text'].apply(lambda x: get_clean(x))")


# In[ ]:





# In[42]:


import spacy
nlp = spacy.load('en_core_web_lg')


# In[43]:


def get_vec(x):
    doc = nlp(x)
    # convert text data to its corresponding numerical vector form
    vec = doc.vector
    return vec


# In[44]:


get_ipython().run_cell_magic('time', '', "df['vec'] = df['text'].apply(lambda x: get_vec(x))")


# In[46]:


X = df['vec'].to_numpy()
X = X.reshape(-1, 1)
X = np.concatenate(np.concatenate(X, axis = 0), axis = 0).reshape(-1, 300)


# In[47]:


y = df['polarity']


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


# In[49]:


clf = LogisticRegression(solver = 'liblinear')


# In[50]:


get_ipython().run_cell_magic('time', '', 'clf.fit(X_train, y_train)')


# In[51]:


y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:





# In[65]:


hyperparameters = [
    {
        'kernel': ['rbf'],
         'gamma': [1e-1, 1e-2],
         'C': [1, 10],
         'degree': [2, 3]
    },
                   
    {
        'kernel': ['linear'],
         'C': [1, 10]
    }
]


# In[ ]:





# In[ ]:





# In[ ]:


scores = ['precision', 'recall']

def run_tuning(model, hyperparameters, scores):
    for score in scores:
        print("Tuning hyperparameters for %s" % score)
        print()
        
        # model creation
        clf = GridSearchCV(model, hyperparameters, scoring='%s_macro' % score, cv = 5, n_jobs = -1)
        
        # mode training
        clf.fit(X_train, y_train)
        
        print('Best parameters set found: ')
        print()
        print(clf.best_params_)
        print()
        
        print('Grid scores in procses: ')
        print()
        means = clf.cv_results_['mean_test_score']
        
        for mean, params in zip(means, clf.cv_results_['params']):
            print('%0.3f for %r' % (mean, params))
            
        print()
        print()
        
        print('Detailed classification report')
        
        #pridicting output using our model
        y_pred = clf.predict(X_test)
        
        # accuracy score
        print(classification_report(y_test, y_pred))


# In[ ]:





# In[96]:


def run_svm(df):
    X = df['text']
    y = df['polarity']

    tfidf = TfidfVectorizer(norm = 'l1', ngram_range=(1,2), analyzer='word', max_features=5000)

    X = tfidf.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0, stratify = y)

    print('shape of X:', X.shape)
    print('')
    
    clf = LogisticRegression(penalty= 'l2', C= 2.7825594022071245, solver = 'liblinear', max_iter= 100)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Printing Report")

    print(classification_report(y_test, y_pred))
    
    # accuracy plot
    plot_confusion_matrix(confusion_matrix(y_test, y_pred))
    plt.show()
    
    return tfidf, clf


# In[97]:


tfidf, clf = run_svm(df)


# In[ ]:





# ## saving & loading trained ml model to local system

# In[54]:


import pickle


# In[55]:


pickle.dump(clf, open('clf.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))


# In[56]:


# deleting modle on jupyter file
del clf
del tfidf


# In[57]:


tfidf


# In[58]:


clf


# In[ ]:





# In[59]:


clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


# In[62]:


x = get_clean("@Ri_Guy that is so bad. thats what happend to tara i was so so excited and then it was a no go  im so sorry")
x


# In[63]:


emotion = clf.predict(tfidf.transform([x]))
emotion


# In[98]:


if emotion[0] == 0:
    print("Negative")
elif emotion[0] == 4:
    print("Positive")


# In[ ]:




