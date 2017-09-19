import numpy as np
import tensorflow as tf
import pandas as pd
import copy
import matplotlib.pyplot as plt

# read in the data set
f = open('./text8/text8')
text8 = []
for line in f.readlines():
    text8.append(line)
text8 = text8[0].split()

voc = {}
for word in text8:
  if word in voc:
    voc[word] += 1
  else:
    voc[word] = 1

voclist = []
for word, count in voc.items():
  temp = [word, count]
  voclist.append(temp)

vocdf = pd.DataFrame(voclist)
ind = np.argsort(vocdf.iloc[:,1])[::-1]
vocdf = vocdf.iloc[ind,:]

finalvoc = list(vocdf.iloc[:9999,0])
finalvoc.append('<UNK>')

text8reduce = copy.copy(text8)
for i,word in enumerate(text8reduce):
  if word in finalvoc:
    text8reduce[i] = finalvoc.index(word)
  else:
    text8reduce[i] = 9999 # set to <UNK>

coocmat = np.zeros(shape=(10000,10000))

windowsize = 3
for i,word in enumerate(text8reduce):
  wordwindow = text8reduce[max(0, i-windowsize):i] + text8reduce[(i+1):min(i+windowsize+1, len(text8reduce))]
  for coword in wordwindow:
    coocmat[word, coword]+=1

embsize = 100
coocmattensor = tf.constant(coocmat)
s, u, v = tf.svd(coocmattensor)

with tf.Session() as sess:
  s, u, v = sess.run([s,u,v])

# visualize most common 100 words using the first two dimensions of the embeddings
for i in range(50):
  fig = plt.gcf()
  fig.set_size_inches(18.5, 10.5)
  plt.text(u[i,0], u[i,1], finalvoc[i])
  plt.xlim((-0.5,0.2))
  plt.ylim((-0.5,0.2))  
plt.savefig('viz.jpg')

emb = u[:, :embsize]
np.save('embedding', emb)
