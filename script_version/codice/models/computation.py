from os import listdir
import string
import pandas as pd
import numpy as np  
import pandas as pd 
import re           
import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import statistics
from models.attention import AttentionLayer
import warnings
import re, string, unicodedata
import nltk
from sklearn.model_selection import train_test_split
import contractions
import inflect
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import tensorflow as tf
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras import backend as K 
from tensorflow.keras.preprocessing.sequence import pad_sequences
lemmatizer = WordNetLemmatizer() 
from rouge import rouge_n_sentence_level
from rouge import rouge_l_sentence_level
from rouge import rouge_n_summary_level
from rouge import rouge_l_summary_level
from rouge import rouge_w_sentence_level
from rouge import rouge_w_summary_level
import pickle
import pickle
import time
import sys

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

def decode_sequence(input_seq, encoder_model, target_word_index, decoder_model, reverse_target_word_index, max_summary_len):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['starttoken']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='endtoken'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'endtoken'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq,target_word_index, reverse_target_word_index ):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['starttoken']) and i!=target_word_index['endtoken']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq, reverse_source_word_index):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

def loss_plot(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

def open_test_data_y():
    return open('../tokenizers_vars/y_tokenizer.pickle', 'rb')

def open_test_data_x():
    return open('../tokenizers_vars/x_tokenizer.pickle', 'rb')

def open_vars():
    return open('../tokenizers_vars/vars.pkl', 'rb')

def inference():
    pass

def computation_bidirectional(x_tr, y_tr, x_val, y_val, y_tokenizer, x_tokenizer, x_voc, y_voc):
    if tf.test.gpu_device_name(): 
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF") 
    max_text_len=300
    max_summary_len=12
    latent_dim = 300
    embedding_dim=100   

    K.clear_session()

    encoder_inputs = Input(shape=(max_text_len,))
    #embedding layer
    enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)
    #encoder lstm 1
    encoder_lstm1 = Bidirectional(LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm1(enc_emb)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Decoder

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    #embedding layer
    dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
    decoder_outputs,_, _, = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

    # Attention layer

    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    #dense layer
    decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model 
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.summary()
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="models/bidirectional/10epochs/saved_model",
                                                                    save_weights_only=False,
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True)

    csv_logger = tf.keras.callbacks.CSVLogger('models/bidirectional/10epochs/training_bidirectional_10.log', append=True)
    print("*"*20, "ARRIVATO QUA", "*"*20)

    history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:],
                  
                  epochs=1, verbose = 1, callbacks=[model_checkpoint_callback, csv_logger],
                  validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))
    
    model.save('models/bidirectional/10epochs/saved_model')

    #history = pd.read_csv('models/bidirectional/10epochs/training_bidirectional_10.log', sep=";")
    loss_plot(history)


def inference_bidirectional(x_test, y_test, x_tokenizer, y_tokenizer):
    
    max_text_len=300
    max_summary_len=12
   
    model = tf.keras.models.load_model('models/bidirectional/10epochs/saved_model')

    print(model.summary())


    reverse_target_word_index=y_tokenizer.index_word
    reverse_source_word_index=x_tokenizer.index_word
    target_word_index=y_tokenizer.word_index

    latent_dim = 300
    embedding_dim=100


    # Encode the input sequence to get the feature vector
    encoder_inputs = model.input[0]   # input_1
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = model.layers[3].output 
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])


    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim*2,), name='dec_st_in_h')
    decoder_state_input_c = Input(shape=(latent_dim*2,), name='dec_st_in_c')
    decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim*2))

    # Get the embeddings of the decoder sequence
    dec_emb_layer = model.layers[4]
    dec_emb2= dec_emb_layer(decoder_inputs) 
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_lstm = model.layers[7]
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    #attention inference
    attn_layer = model.layers[8]
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_dense = model.layers[10]
    decoder_outputs2 = decoder_dense(decoder_inf_concat) 

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])
    
    for i in range(0,1):
        print("Review:",seq2text(x_test[i],reverse_source_word_index))
        print("Original summary:",seq2summary(y_test[i], target_word_index , reverse_target_word_index))
        print("Predicted summary:",decode_sequence(x_test[i].reshape(1,max_text_len), encoder_model, target_word_index, decoder_model, reverse_target_word_index, max_summary_len))
        print("\n")

    original_text = []
    original_summary = []
    created_summary = []

    for i in range(0,2000):
        #barra caricamento
        progress(i, 2000, status='Doing very long job')

        original_text.append(seq2text(x_test[i] ,reverse_source_word_index))
        original_summary.append(seq2summary(y_test[i], target_word_index, reverse_target_word_index))
        created_summary.append(decode_sequence(x_test[i].reshape(1,max_text_len), encoder_model, target_word_index, decoder_model, reverse_target_word_index, max_summary_len))
    
    results = pd.DataFrame()
    results["Original_text"] = original_text
    results["Original_summary"] = original_summary
    results["Created_summary"] = created_summary

    results.to_csv("models/bidirectional/10epochs/results_predictions_bi_10.csv")

    # Compute il rouge
    results=pd.read_csv("models/bidirectional/10epochs/results_predictions_bi_10.csv")
    results["Created_summary"].replace(np.nan, 'NaN', inplace=True)
    reference_sentences = results["Original_summary"].to_list()
    summary_sentences = results["Created_summary"].to_list()

    list_rouge_r2 = []
    list_recall_r2 = []
    list_precision_r2 = []
    list_rouge_r1= []
    list_recall_r1 = []
    list_precision_r1 = []

    for i in range(0, len(reference_sentences)):
        #barra caricamento
        progress(i, len(reference_sentences), status='Doing very long job')
        # My long python program here   

        reference_sentence = reference_sentences[i].split()
        summary_sentence = summary_sentences[i].split()
        
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

    mean_rouge_r2 = statistics.mean(list_rouge_r2)  
    mean_rouge_r1 = statistics.mean(list_rouge_r1)  

    print("Mean ROUGE-2: ", mean_rouge_r2)
    print("Mean ROUGE-1: ", mean_rouge_r1)

def computation_monodirectional():
    max_text_len=300
    max_summary_len=12

    K.clear_session()

    latent_dim = 300
    embedding_dim=100

    # Encoder

    encoder_inputs = Input(shape=(max_text_len,))
    #embedding layer
    enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)
    #encoder lstm 1
    encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c = encoder_lstm1(enc_emb)

    # Decoder

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    #embedding layer
    dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

    # Attention layer

    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    #dense layer
    decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model 
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.summary()
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/monodirectional/10epochs/training_monodirectional_10.log",
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

    csv_logger = tf.keras.callbacks.CSVLogger("models/monodirectional/10epochs/training_monodirectional_10.log", append=True)
    history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,
                  epochs=10, verbose = 1, callbacks=[model_checkpoint_callback, csv_logger],
                  validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

    model.save("models/monodirectional/10epochs/saved_model")
    loss_plot(history)

def inference_monodirectional(x_test, y_test, x_tokenizer, y_tokenizer):
    max_text_len=300
    max_summary_len=12

    model = tf.keras.models.load_model("models/monodirectional/10epochs/saved_model")

    reverse_target_word_index=y_tokenizer.index_word
    reverse_source_word_index=x_tokenizer.index_word
    target_word_index=y_tokenizer.word_index


    latent_dim = 300
    embedding_dim=100

    # Encode the input sequence to get the feature vector
    encoder_inputs = model.input[0]   # input_1
    encoder_outputs, state_h, state_c = model.layers[4].output 
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim,), name='dec_st_in_h')
    decoder_state_input_c = Input(shape=(latent_dim,), name='dec_st_in_c')
    decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))

    # Get the embeddings of the decoder sequence
    dec_emb_layer = model.layers[3]
    dec_emb2= dec_emb_layer(decoder_inputs) 
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_lstm = model.layers[5]
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    #attention inference
    attn_layer = model.layers[6]
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_dense = model.layers[8]
    decoder_outputs2 = decoder_dense(decoder_inf_concat) 

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])
    
    for i in range(0,1):
        print("Review:",seq2text(x_test[i], reverse_source_word_index))
        print("Original summary:",seq2summary(y_test[i],target_word_index, reverse_target_word_index))
        print("Predicted summary:",decode_sequence(x_test[i].reshape(1,max_text_len), encoder_model, target_word_index, decoder_model, reverse_target_word_index, max_summary_len))
        print("\n")
    
    original_text = []
    original_summary = []
    created_summary = []

    for i in range(0,2000):
        #barra caricamiento
        progress(i, 2000, status='Doing very long job')

        original_text.append(seq2text(x_test[i], reverse_source_word_index))
        original_summary.append(seq2summary(y_test[i], target_word_index, reverse_target_word_index))
        created_summary.append(decode_sequence(x_test[i].reshape(1,max_text_len), encoder_model, target_word_index, decoder_model, reverse_target_word_index, max_summary_len))

    results = pd.DataFrame()
    results["Original_text"] = original_text
    results["Original_summary"] = original_summary
    results["Created_summary"] = created_summary

    results.to_csv("models/monodirectional/10epochs/results_predictions_mono_10.csv")

    results=pd.read_csv("models/monodirectional/10epochs/results_predictions_mono_10.csv")
    results["Created_summary"].replace(np.nan, 'NaN', inplace=True)

    reference_sentences = results["Original_summary"].to_list()
    summary_sentences = results["Created_summary"].to_list()

    list_rouge_r2 = []
    list_recall_r2 = []
    list_precision_r2 = []
    list_rouge_r1= []
    list_recall_r1 = []
    list_precision_r1 = []

    for i in range(0, len(reference_sentences)):
        #barra di frate maronno
        progress(i, len(reference_sentences), status='Doing very long job')

        reference_sentence = reference_sentences[i].split()
        summary_sentence = summary_sentences[i].split()
        
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

    mean_rouge_r2 = statistics.mean(list_rouge_r2)  
    mean_rouge_r1 = statistics.mean(list_rouge_r1)  

    print("Mean ROUGE-2: ", mean_rouge_r2)
    print("Mean ROUGE-1: ", mean_rouge_r1)


    
def computation(model):
    x_tr = np.load("../final_data/x_tr.npy")
    y_tr = np.load("../final_data/y_tr.npy")
    x_val = np.load("../final_data/x_val.npy")
    y_val = np.load("../final_data/y_val.npy")
    x_test = np.load("../final_data/x_test.npy")
    y_test = np.load("../final_data/y_test.npy")

    with open_test_data_y() as f:
        y_tokenizer = pickle.load(f) 

    with open_test_data_x() as f:
        x_tokenizer = pickle.load(f) 

    with open_vars() as f: 
        x_voc, y_voc = pickle.load(f)

    with open_test_data_y() as f:
        y_tokenizer = pickle.load(f) 

    with open_test_data_x() as f:
        x_tokenizer = pickle.load(f)


    if model == "bidirectional_model":
        if not os.path.exists('models/bidirectional/10epochs'):
            computation_bidirectional(x_tr, y_tr, x_val, y_val, y_tokenizer, x_tokenizer, x_voc, y_voc)

        print("*"*20, "Training finito", "*"*20)
        inference_bidirectional(x_test, y_test, x_tokenizer, y_tokenizer)
        print("*"*20, "Inferenza finita", "*"*20)

    else:
        
        if not os.path.exists('models/monodirectional/10epochs'):
            computation_monodirectional(x_tr, y_tr, x_val, y_val, y_tokenizer, x_tokenizer, x_voc, y_voc)

        print("*"*20, "Training finito", "*"*20)
        inference_monodirectional(x_test, y_test, x_tokenizer, y_tokenizer)
        print("*"*20, "Inferenza finita", "*"*20)

    