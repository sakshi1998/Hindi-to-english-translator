import gensim
import numpy as np
import matplotlib.pyplot as plt
 
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings
warnings.filterwarnings(action = 'ignore') 


from sklearn.manifold import TSNE

sample = open(r"C:\Users\Himani khurana\Documents\MAJOR_project\inputengsentences.txt", "r") 
s = sample.read() 
  
# Replaces escape character with space 
f = s.replace("\n", " ")


data = [] 
  
# iterate through each sentence in the file 
for i in sent_tokenize(f): 
    temp = [] 
     
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
         temp.append(j.lower())
        
  
    data.append(temp) 
  
# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = 100, window = 5) 
model = gensim.models.KeyedVectors.load_word2vec_format(data)


# Test the loaded word2vec model in gensim
# We will need the raw vector for a word
print(model['fresh']) 

# We will also need to get the words closest to a word
model.similar_by_word('fresh')

def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()


display_closestwords_tsnescatterplot(model, 'teeth')    
