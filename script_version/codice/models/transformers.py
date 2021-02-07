from os import listdir
import string
import pandas as pd
import numpy as np  
import pandas as pd 
from IPython.display import display, clear_output
import re           
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
import pickle
import inflect
from bs4 import BeautifulSoup
import tensorflow as tf
from rouge import rouge_n_sentence_level
from rouge import rouge_l_sentence_level
from rouge import rouge_n_summary_level
from rouge import rouge_l_summary_level
from rouge import rouge_w_sentence_level
from rouge import rouge_w_summary_level
import statistics
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration,  BartTokenizer, PegasusForConditionalGeneration, PegasusTokenizer
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


def evaluation_pretrained():
    results=pd.read_csv("../data_transformer/results_predictions_transformers.csv")
    reference_sentences = results["Original_summary"].to_list()
    summary_sentences_t5 = results["Created_summary_t5"].to_list()
    summary_sentences_bart = results["Created_summary_bart"].to_list()

    mean_rouge_r2_t5 = 0
    mean_rouge_r1_t5= 0
    mean_rouge_r2_bart= 0
    mean_rouge_r1_bart= 0

    summary_sentences_list = [summary_sentences_t5, summary_sentences_bart]
    mean_rouge_r2_list = [mean_rouge_r2_t5, mean_rouge_r2_bart]
    mean_rouge_r1_list = [mean_rouge_r1_t5, mean_rouge_r1_bart]

for iteration in range(2):
    print("Model: ", iteration)
    
    list_rouge_r2 = []
    list_recall_r2 = []
    list_precision_r2 = []
    list_rouge_r1= []
    list_recall_r1 = []
    list_precision_r1 = []

    for i in range(0, len(reference_sentences)):
        clear_output(wait=True)
        print(i)
        reference_sentence = reference_sentences[i].split()
        summary_sentence = summary_sentences_list[iteration][i].split()
        
        # Calculate ROUGE-2.
        recall_r2, precision_r2, rouge_r2 = rouge_n_sentence_level(summary_sentence, reference_sentence, 2)

        list_rouge_r2.append(rouge_r2)
        list_recall_r2.append(recall_r2)
        list_precision_r2.append(precision_r2)

        # Calculate ROUGE-1.
        recall_r1, precision_r1, rouge_r1 = rouge_n_sentence_level(summary_sentence, reference_sentence, 1)

        list_rouge_r1.append(rouge_r1)
        list_recall_r1.append(recall_r1)
        list_precision_r1.append(precision_r1)

    mean_rouge_r2_list[iteration] = statistics.mean(list_rouge_r2)  
    mean_rouge_r1_list[iteration] = statistics.mean(list_rouge_r1)  

    print("Mean ROUGE-2 : ", mean_rouge_r2_list[iteration])
    print("Mean ROUGE-1 : ", mean_rouge_r1_list[iteration])

    print("Model T5")
    print("Rouge 2: ", mean_rouge_r2_list[0], "Rouge 1: ",mean_rouge_r1_list[0])
    print("Model BART")
    print("Rouge 2: ", mean_rouge_r2_list[1], "Rouge 1: ",mean_rouge_r1_list[1])

def pretrained():
    if tf.test.gpu_device_name(): 
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

# Load data
    data = pd.read_csv("../data_transformer/test_dataset.csv")
    data["cleaned_highlight"] = data["cleaned_highlight"].str.replace("starttoken", "")
    data["cleaned_highlight"] = data["cleaned_highlight"].str.replace("endtoken", "")

    stories = data["cleaned_text"]
    summaries = data["cleaned_highlight"]

    # initialize the model architecture and weights
    model_t5 = T5ForConditionalGeneration.from_pretrained("t5-base")
    model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    # initialize the model tokenizer
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base")
    tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-base")

    original_text = []
    original_summary = []
    created_summary_t5 = []
    created_summary_bart = []
    created_summary_pegasus = []

    for i in range(0,2000):
        clear_output(wait=True)
        print(i)
        original_text.append(stories[i])
        original_summary.append(summaries[i])
        #t5
        inputs_t5 = tokenizer_t5.encode("summarize: " + stories[i], return_tensors="pt", max_length=300, truncation=True)
        outputs_t5 = model_t5.generate(
            inputs_t5, 
            max_length=12, 
            min_length=3, 
            length_penalty=1.0,  
            early_stopping=True)
        created_summary_t5.append(tokenizer_t5.decode(outputs_t5[0]))
        #bart
        inputs_bart = tokenizer_bart.encode(stories[i], return_tensors="pt", max_length=300, truncation=True)
        outputs_bart = model_bart.generate(
            inputs_bart, 
            max_length=12, 
            min_length=3, 
            length_penalty=1.0,  
            early_stopping=True)
        created_summary_bart.append(tokenizer_bart.decode(outputs_bart[0]))

    results = pd.DataFrame()
    results["Original_text"] = original_text
    results["Original_summary"] = original_summary
    results["Created_summary_t5"] = created_summary_t5
    results["Created_summary_bart"] = created_summary_bart
    results["Created_summary_t5"] = results["Created_summary_t5"].str.replace("<pad> ", "")
    results["Created_summary_bart"] = results["Created_summary_bart"].str.replace("</s><s>", "")
    results["Created_summary_bart"] = results["Created_summary_bart"].str.replace("</s>", "")

    results.to_csv("../data_transformer/results_predictions_transformers.csv")

def transformers_execution():
    if not os.path.exists('../data_transformer/results_predictions_transformers'):    
        pretrained()
    evaluation_pretrained()
    