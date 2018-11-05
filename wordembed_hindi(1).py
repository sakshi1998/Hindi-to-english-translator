
# Python program to generate word vectors using Word2Vec 
# -*- coding: utf-8 -*-  
# importing all necessary modules 
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
  
warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec 
  
#  Reads ‘inputenglishsentences.txt’ file 
sample = open(r"C:\Users\Himani khurana\Documents\MAJOR_project\hindi sentences.txt", "r") 
s = sample.read() 
  
# Replaces escape character with space 
f = s.replace("\n", " ")

#stop words removal
import nltk
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
  
data = [] 
  
# iterate through each sentence in the file 
for i in sent_tokenize(f): 
    temp = [] 
     
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        if j not in stop_words:
         temp.append(j.lower())
        
  
    data.append(temp)
print(data)
    
  
# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = 100, window = 5) 
  

  
# Create Skip Gram model 
model = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
                                             window = 5, sg = 1) 
  

from sklearn.decomposition import PCA
from matplotlib import pyplot

#PCA Skip-Gram
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()


#PCA CBOW
Y = model1[model1.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(Y)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model1.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()


