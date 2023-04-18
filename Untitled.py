#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# In[11]:


get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install pandas')
get_ipython().system('pip install sentence-transformers')
get_ipython().system('pip install scikit-learn')


# In[17]:


model = SentenceTransformer('jhgan/ko-sroberta-multitask')

sentences = ["안녕하세요?", "한국어 문장 임베딩을 위한 버트 모델입니다."]
embeddings = model.encode(sentences)

print(embeddings)


# In[18]:


df = pd.read_csv('wellness_dataset_original.csv')

df.head()


# In[19]:


df = df.drop(columns=['Unnamed: 3'])

df.head()


# In[20]:


df = df[~df['챗봇'].isna()]

df.head()


# In[21]:


df.loc[0, '유저']


# In[22]:


model.encode(df.loc[0, '유저'])


# In[23]:


df['embedding'] = pd.Series([[]] * len(df)) # dummy

df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))

df.head()


# In[24]:


df.to_csv('wellness_dataset.csv', index=False)


# In[25]:


text = '요즘 머리가 아프고 너무 힘들어'

embedding = model.encode(text)


# In[26]:


df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())

df.head()


# In[27]:


answer = df.loc[df['distance'].idxmax()]

print('구분', answer['구분'])
print('유사한 질문', answer['유저'])
print('챗봇 답변', answer['챗봇'])
print('유사도', answer['distance'])


# In[ ]:




