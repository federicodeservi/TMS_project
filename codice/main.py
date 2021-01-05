'''
Main module to execute all the steps of the project with the following command line parameters:
'''
from preprocessing.preprocessing import preprocessing_data

from models.computation import inference
import time
import pandas as pd
import os



pd.options.mode.chained_assignment = None


def main():
    '''
    Main function that execute computation
    '''
    if not os.path.exists('../final_data/x_tr.npy'):
        preprocessing_data()
        inference()
    else:
        inference()
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time \n--- %s s ---" % (time.time() - start_time))