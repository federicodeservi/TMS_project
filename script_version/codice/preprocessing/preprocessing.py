from os import listdir
import string
import pandas as pd
import numpy as np  
import pandas as pd 
import re           
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import re, string, unicodedata
import nltk
from sklearn.model_selection import train_test_split
import contractions
import inflect
from bs4 import BeautifulSoup
import tensorflow as tf
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
lemmatizer = WordNetLemmatizer() 
import pickle

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


def load_doc(filename):
	file = open(filename, encoding='utf-8')
	text = file.read()
	file.close()
	return text

# split a document into news story and highlights
def split_story(doc):
	# find first highlight
	index = doc.find('@highlight')
	# split into story and highlights
	story, highlights = doc[:index], doc[index:].split('@highlight')
	# strip extra white space around each highlight
	highlights = [h.strip() for h in highlights if len(h) > 0]
	return story, highlights

# load all stories in a directory
def load_stories(directory):
    stories = []
    highlights =[]
    for name in listdir(directory)[:10000]:
        filename = directory + '/' + name
		# load document
        doc = load_doc(filename)
		# split into story and highlights
        story, highlight = split_story(doc)
		# store
        stories.append(story)
        highlights.append(highlight)
    data = pd.DataFrame()
    data["story"] = stories
    data["highlight"] = highlights
    
    return  data

def remove_non_ascii(words):
    """Remove non-ASCII characters"""
    new_words = unicodedata.normalize('NFKD', words).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase"""
    new_words = words.lower()
    return new_words

def remove_punctuation(words):
    """Remove punctuation"""
    new_words = re.sub(r'\([^)]*\)', '', words)
    return new_words

def replace_numbers(words):
    """Replace all integer occurrences"""
    new_words = re.sub("[^a-zA-Z]", " ", words) 
    return new_words

def remove_stopwords(words):
    """Remove stop words"""
    stop_words = set(stopwords.words('english')) 
    new_words = [w for w in words.split() if not w in stop_words]
    long_words=[]
    for i in new_words:
        if len(i)>=1:                  
            long_words.append(i)   

    return (" ".join(long_words)).strip()


def replace_contractions(words):
    """Replace contractions in string of text"""
    return contractions.fix(words)

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = replace_contractions(words)
    words = remove_stopwords(words)

    return words

def lemmatize(words):
    words = lemmatizer.lemmatize(words)
    return words

def preprocessing_data():
    
    directory = "../dati/cnn/stories"
    data = load_stories(directory)
    print("*************** IMPORT THE DATA ***************")
    print('Loaded Stories %d' % len(data))
    data = data.explode("highlight")

    print("*************** NORMALIZATION ***************")
    stop_words = set(stopwords.words('english')) 

    cleaned_story = []
    for t in data['story']:
        cleaned_story.append(normalize(t))

    cleaned_highlight = []
    for t in data['highlight']:
        cleaned_highlight.append(normalize(t))

    data['normalized_text']=cleaned_story
    data['normalized_highlight']=cleaned_highlight
    data['normalized_highlight'].replace('', np.nan, inplace=True)
    data.dropna(axis=0,inplace=True)

    print("*************** LEMMATIZATION ***************")
    lemmatized_story = []
    for t in data['normalized_text']:
        lemmatized_story.append(lemmatize(t))
        
    lemmatized_highlight = []
    for t in data['normalized_highlight']:
        lemmatized_highlight.append(lemmatize(t))

    data['cleaned_text']=lemmatized_story
    data['cleaned_highlight']=lemmatized_highlight
    data['cleaned_highlight'].replace('', np.nan, inplace=True)
    data.dropna(axis=0,inplace=True)

    data['cleaned_highlight'] = data['cleaned_highlight'].apply(lambda x : 'starttoken '+ x + ' endtoken')
    data = data[["cleaned_text", "cleaned_highlight"]]

    x_tr,x_val,y_tr,y_val=train_test_split(data['cleaned_text'],data['cleaned_highlight'],test_size=0.1,random_state=0,shuffle=True)
    
    print("*************** TOKENIZATION ***************")
    x_tokenizer = Tokenizer() 
    x_tokenizer.fit_on_texts(list(x_tr))

    thresh=4

    cnt=0
    tot_cnt=0
    freq=0
    tot_freq=0

    for key,value in x_tokenizer.word_counts.items():
        tot_cnt=tot_cnt+1
        tot_freq=tot_freq+value
        if(value<thresh):
            cnt=cnt+1
            freq=freq+value
    
    print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
    print("Total Coverage of rare words:",(freq/tot_freq)*100)

    print("*************** PLOT ***************")
    text_word_count = []
    summary_word_count = []

    # populate the lists with sentence lengths
    for i in data['cleaned_text']:
        text_word_count.append(len(i.split()))

    for i in data['cleaned_highlight']:
        summary_word_count.append(len(i.split()))

    length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})
    length_df.hist(bins = 30)
    plt.show()

    cnt=0
    for i in data['cleaned_highlight']:
        if(len(i.split())<=14):
            cnt=cnt+1
    print(cnt/len(data['cleaned_highlight']))

    max_text_len=300
    max_summary_len=12
    print("*************** INPUT TOKENIZATION ***************")
    #prepare a tokenizer for reviews on training data
    x_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
    x_tokenizer.fit_on_texts(list(x_tr))

    #convert text sequences into integer sequences
    x_tr_seq = x_tokenizer.texts_to_sequences(x_tr) 
    x_val_seq = x_tokenizer.texts_to_sequences(x_val)
    #padding zero upto maximum length
    x_tr = pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
    x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

    #size of vocabulary ( +1 for padding token)
    x_voc = x_tokenizer.num_words + 1


    print("*************** TARGET TOKENIZATION ***************")
    #prepare a tokenizer for reviews on training data
    y_tokenizer = Tokenizer()   
    y_tokenizer.fit_on_texts(list(y_tr))

    thresh=6

    cnt=0
    tot_cnt=0
    freq=0
    tot_freq=0

    for key,value in y_tokenizer.word_counts.items():
        tot_cnt=tot_cnt+1
        tot_freq=tot_freq+value
        if(value<thresh):
            cnt=cnt+1
            freq=freq+value
        
    print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
    print("Total Coverage of rare words:",(freq/tot_freq)*100)
  

    #prepare a tokenizer for reviews on training data
    y_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
    y_tokenizer.fit_on_texts(list(y_tr))

    #convert text sequences into integer sequences
    y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr) 
    y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) 

    #padding zero upto maximum length
    y_tr    =   pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
    y_val   =   pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

    #size of vocabulary
    y_voc  =   y_tokenizer.num_words +1

    y_tokenizer.word_counts['starttoken'],len(y_tr)

    ind=[]
    for i in range(len(y_tr)):
        cnt=0
        for j in y_tr[i]:
            if j!=0:
                cnt=cnt+1
        if(cnt==2):
            ind.append(i)

    y_tr=np.delete(y_tr,ind, axis=0)
    x_tr=np.delete(x_tr,ind, axis=0)

    ind=[]
    for i in range(len(y_val)):
        cnt=0
        for j in y_val[i]:
            if j!=0:
                cnt=cnt+1
        if(cnt==2):
            ind.append(i)

    y_val=np.delete(y_val,ind, axis=0)
    x_val=np.delete(x_val,ind, axis=0)

    print("*************** SAVE THE FILES ***************")
    np.save("../final_data/x_tr.npy", x_tr)
    np.save("../final_data/y_tr.npy", y_tr)
    np.save("../final_data/x_val.npy", x_val)
    np.save("../final_data/y_val.npy", y_val)


    # saving
    with open('../tokenizers_vars/y_tokenizer.pickle', 'wb') as handle:
    pickle.dump(y_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../tokenizers_vars/x_tokenizer.pickle', 'wb') as handle:
        pickle.dump(x_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../tokenizers_vars/vars.pkl', 'wb') as f:  
        pickle.dump([x_voc, y_voc], f)
    
    return x_voc, y_voc
    