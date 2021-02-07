'''
Main module to execute all the steps of the project with the following command line parameters:
'''
from preprocessing.preprocessing_train import preprocessing_data
from preprocessing.preprocessing_test import preprocessing_data_test

from models.computation import computation
from models.computation import inference
from models.transformers import transformers_execution
import time
import pandas as pd
import os
import argparse


pd.options.mode.chained_assignment = None

def mkdir_p(mypath):
    '''
    Create directories that are used if they not exist
    Parameters:
        mypath  -Required:  output path chosen
    '''
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def main():
    '''
    Main function that execute computation
    '''


    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=False, default="bidirectional_model",
    help="Insert a tipe of model between 'bidirectional' or 'monodirectional'!")
    args = parser.parse_args()

    mkdir_p("../final_data")

    # Crea le directory necessarie se non esistono

    if not os.path.exists('models/bidirectional'):
        mkdir_p('models/bidirectional')
    
    if not os.path.exists('models/bidirectional/10epochs'):
        mkdir_p('models/bidirectional/10epochs')
    
    if not os.path.exists('models/bidirectional/20epochs'):
        mkdir_p('models/bidirectional/20epochs')

    if not os.path.exists('models/monodirectional'):
        mkdir_p('models/monodirectional')
        
    if not os.path.exists('models/monodirectional/10epochs'):
        mkdir_p('models/monodirectional/10epochs')
    
    if not os.path.exists('models/monodirectional/20epochs'):
        mkdir_p('models/monodirectional/20epochs')


    # Salta il preprocessing se è già stato fatto
    if not os.path.exists('../final_data/x_tr.npy'):    
        preprocessing_data()
        
    if not os.path.exists('../final_data/x_test.npy'):   
        preprocessing_data_test()

    computation(args.model)
    transformers_execution()



if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time \n--- %s s ---" % (time.time() - start_time))