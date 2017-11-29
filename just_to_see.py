
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


res = pd.read_csv('predictions.txt',sep = ' ', header = None)
res = res.T
results  = res.loc[:,0:19]


# In[4]:


##Анчик эта ячейка для того, чтобы записать все в файл, понял, бля???
data_init = pd.read_csv('dm-final-testdist.txt', header = None)
data  = pd.concat([data_init, results], axis=1, ignore_index=True)
data.to_csv('final_results.csv', sep=',', header=None, index=False)


# In[5]:


data.shape


# In[6]:


### А эти две чтобы плотить
def get_data(file_name, batch):
   data = pd.read_csv(file_name, header=None)
   
   train = data.loc[:, :]
   test = data.loc[:, :]
   
   train_X = np.expand_dims((train.loc[:, 1:batch]).T, axis=2)
   train_Y = np.expand_dims((train.loc[:, (batch+1):]).T, axis=2)

   test_X = np.expand_dims((test.loc[:, 1:batch]).T, axis=2)
   test_Y = np.expand_dims((test.loc[:, (batch+1):]).T, axis=2)
   
   return train_X, train_Y, test_X, test_Y
train_X, train_Y, test_X, test_Y = get_data('final_results.csv', 100)


# In[7]:


plt.plot(range(len(train_X[:, 0,:])), train_X[:,300,:], "o--b")
plt.plot(range(len(train_X[:, 0,:]), len(train_X[:, 0,:])+len(train_Y[:, 0,:])), train_Y[:,300,:], "x--r")
plt.show()


# In[ ]:





# In[ ]:




