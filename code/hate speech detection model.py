#!/usr/bin/env python
# coding: utf-8

# # **Hate Speech Classification**

# Hate speech is defined by the Cambridge Dictionary as "public speech that expresses hate or encourages violence towards a person or group based on something such as race, religion, sex, or sexual orientation".

# Steps to classify hate speech
# - Preprocess the text data
# - Convert text to numerical tokens
# - Build and Train ML
# - Test the Model
# - Save and use it later

# In[1]:


#!pip install git+https://github.com/PraNjaL-16/preprocess_pranjalSagar.git


# In[2]:


#!pip install spacy


# In[3]:


# for conversion of text data to numerical representation
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Dropout
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalMaxPooling1D, MaxPooling1D


# In[4]:


import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
import preprocess_pranjalSagar as ps


# In[5]:


# df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/hate_speech_dataset/master/data.csv')
# df


# In[6]:


df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/hate_speech_dataset/master/data.csv', index_col = 0)
df


# In[7]:


# class is target variable
# class -> 0: hate speech, 1: offensive language, 2: neither


# In[8]:


df['class'].value_counts()


# In[9]:


# balancing dataset
vc = df['class'].value_counts()
index = list(vc.index)
count = min(vc.values)
count, index


# In[ ]:





# In[10]:


# extract only thoes data which have class == 1
df[df['class'] == 1]


# In[11]:


# randomly extract five data sample having class == 1
df[df['class'] == 1].sample(5)


# In[ ]:





# In[12]:


# balancing dataset
df_bal = pd.DataFrame()
for i in index:
  temp = df[df['class'] == i].sample(count)
  df_bal = df_bal.append(temp, ignore_index = True)


# In[13]:


df_bal


# In[14]:


# balanced dataset having equal no. of sample data for all the class
df_bal['class'].value_counts()


# In[15]:


df = df_bal.copy()


# In[16]:


# balanced dataset having equal no. of sample data for all the class
df['class'].value_counts()


# In[ ]:





# ## **Data preprocessing**

# In[17]:


def get_clean(x):
  x = str(x).lower().replace('\\', '').replace('_', '')
  x = ps.cont_exp(x)
  x = ps.remove_emails(x)
  x = ps.remove_urls(x)
  x = ps.remove_html_tags(x)
  x = ps.remve_rt(x)
  x = ps.remove_accented_chars(x)
  x = ps.remove_special_chars(x)
  # looovvveeee -> love
  x = re.sub("(.)\\1{2,}", "\\1", x)
  return x


# In[ ]:





# In[18]:


x = 'iiii _#@ youuuuuuuuu'
x = get_clean(x)
x


# In[ ]:





# In[19]:


df['tweet'] = df['tweet'].apply(lambda x: get_clean(x))


# In[20]:


df


# In[ ]:





# ## **Text tokenization**

# In[21]:


type(df['tweet'])


# In[22]:


text = df['tweet'].tolist()


# In[23]:


type(text)


# In[24]:


text[:2]


# In[25]:


# for conversion of text data to numerical representation
token = Tokenizer()
token.fit_on_texts(text)


# In[26]:


# unique words present in the dataset
len(token.word_counts)


# In[27]:


print(token.index_word)


# In[28]:


# token.word_counts


# In[29]:


# conversion of text data to numerical representation
x = ['i love you']
token.texts_to_sequences(x)


# In[30]:


vocab_size = len(token.word_counts) + 1


# In[31]:


# conversion of text data to numerical representation
encoded_text = token.texts_to_sequences(text)


# In[32]:


print(encoded_text)


# In[33]:


# fixing length of input
max_length = 120
X = pad_sequences(encoded_text, maxlen=max_length, padding = 'post')


# In[34]:


print(X)


# In[ ]:





# # **train test split**

# In[35]:


from keras.utils import np_utils


# In[36]:


y = df['class']


# In[37]:


y


# In[38]:


# one hot encoded target variable
y = np_utils.to_categorical(df['class'])


# In[39]:


y


# In[ ]:





# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


# In[41]:


X.shape, y.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:





# In[ ]:





# ## **Build & train CNN model**

# In[42]:


from tensorflow.keras.optimizers import Adam


# In[43]:


# internally all the tokens will be converted to vector representaion
vec_size = 300

# model creation
model = Sequential()
model.add(Embedding(vocab_size, vec_size, input_length=max_length))

model.add(Conv1D(32, 2, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(3, activation='softmax'))


# In[44]:


# model compilation
model.compile(optimizer=Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# model training
model.fit(X_train, y_train, epochs = 2, validation_data=(X_test, y_test), shuffle = True)


# In[ ]:





# ## **Model testing**

# In[45]:


from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


# In[46]:


# predicting output using the Model
# y_pred = model.predict_classes(X_test)
y_pred = np.argmax(model.predict(X_test), axis=-1)


# In[47]:


# making y_test & y_pred to same dimensions
y_tests = np.argmax(y_test, axis=-1)


# In[48]:


plot_confusion_matrix(confusion_matrix(np.argmax(y_test, axis=-1), y_pred))


# In[49]:


print(classification_report(np.argmax(y_test, axis=-1), y_pred))


# In[ ]:





# ## **testing model with custom data**

# In[50]:


x = 'hey dude whass up'


# In[51]:


def get_encoded(x):
  x = get_clean(x)

  # convering text data to numerical sequence
  x = token.texts_to_sequences([x])

  # fixing input data length
  x = pad_sequences(x, maxlen=max_length, padding='post')

  return x


# In[52]:


x = get_encoded(x)
x


# In[53]:


# pedcitng output using our model
np.argmax(model.predict(x), axis=-1)


# In[ ]:





# In[54]:


x = 'hey bitch whass up'

# pedcitng output using our model
np.argmax(model.predict(get_encoded(x)), axis=-1)


# In[ ]:





# ## **storing model to local computer**

# In[59]:


from tensorflow.keras.models import load_model


# In[64]:


model.save('hate_Speech_model', save_format='h5')


# In[65]:


import pickle


# In[66]:


pickle.dump(token, open('token.pkl', 'wb'))


# In[67]:


del model


# In[68]:


del token


# In[69]:


model


# In[70]:


token


# In[71]:


model = load_model('hate_Speech_model')


# In[72]:


token = pickle.load(open('token.pkl', 'rb'))


# In[73]:


x = 'hey bitch whass up'

# pedcitng output using our model
np.argmax(model.predict(get_encoded(x)), axis=-1)


# In[ ]:





# In[ ]:




