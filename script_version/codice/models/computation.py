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
from models.attention import AttentionLayer
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
from tensorflow.keras import backend as K 
from tensorflow.keras.preprocessing.sequence import pad_sequences
lemmatizer = WordNetLemmatizer() 
import pickle


def decode_sequence(input_seq):
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


def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['starttoken']) and i!=target_word_index['endtoken']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

def computation(x_voc, y_voc):
    max_text_len=300
    max_summary_len=12

    x_tr = np.load('../final_data/x_tr.npy')
    x_val = np.load('../final_data/x_val.npy')
    y_tr = np.load('../final_data/y_tr.npy')
    y_val = np.load('../final_data/y_val.npy')

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
    history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,
                  epochs=10, verbose = 1,
                  validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))
    
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

    model.save_weights("weights.h5")

def inference():

    max_text_len=300
    max_summary_len=12

    #if not os.path.exists('weights.h5'):
    #    computation(x_voc, y_voc)
    #else:
    model = load("weights.h5")
    reverse_target_word_index=y_tokenizer.index_word
    reverse_source_word_index=x_tokenizer.index_word
    target_word_index=y_tokenizer.word_index

    # Encode the input sequence to get the feature vector
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))

    # Get the embeddings of the decoder sequence
    dec_emb2= dec_emb_layer(decoder_inputs) 
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    #attention inference
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = decoder_dense(decoder_inf_concat) 

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])

    for i in range(0,30):
        print("Review:",seq2text(x_tr[i]))
        print("Original summary:",seq2summary(y_tr[i]))
        print("Predicted summary:",decode_sequence(x_tr[i].reshape(1,max_text_len)))
        print("\n")