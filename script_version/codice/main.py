'''
Main module to execute all the steps of the project with the following command line parameters:
'''
from preprocessing.preprocessing import preprocessing_data

from models.computation import computation
from models.computation import inference
import time
import pandas as pd
import os



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
    '''
    if not os.path.exists('../final_data/x_tr.npy'):
        x_voc, y_voc = preprocessing_data()
        computation(x_voc, y_voc)
        inference(x_voc, y_voc)
    else:
        inference(x_voc, y_voc)
    '''
    mkdir_p("../final_data")
    x_voc, y_voc =  preprocessing_data()
    computation(x_voc, y_voc)
    inference()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time \n--- %s s ---" % (time.time() - start_time))