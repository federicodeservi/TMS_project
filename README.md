# Text Mining and Search project

The aim of this project is to generate summaries for news articles taken from the CNN dataset, through a technique called abstractive text summarization. This involves creating models that generate structurally correct and meaningful summaries from the text. Two different models are created and trained for 20 epochs, to see the results that such models could produce when trained on a highend consumer machine. The performance assessed in terms of ROUGE is low, and this confirms the training complexity of Seq2Seq models. However, some differences in terms of fluency between the two models are found. Also, it is known that in the context of text summarization, quantitative measures must always be accompanied by a human evaluation. After analyzing the predictions of both models it can be said that the bidirectional model is the one that produces lexically more complex results. Performance was then compared with pretrained models such as BART and T5 in order to evaluate
the differences.

## Preparation

We suggest to install all the useful libraries in a virtual environment in order to have the same version and to not create confilcts and warnings.

First of all in a command line:

`virtualenv env`

Then activate it

`source env/bin/activate`

Then install all the libraries used with:

`pip install -r requirements.txt`

If some of the packages are already installed in the PC this operation is faster. ROUGE library must be installed separately because it has some dependencies. In particular:

`git clone https://github.com/neural-dialogue-metrics/rouge.git`

`python ./setup.py install`

There can be some dependecies that must be installed due to the version of python, in this case

`pip install git+https://github.com/tagucci/pythonrouge.git`

`python -m doctest ./rouge/*.py -v`

In order to install **contractions** you must also install 

https://visualstudio.microsoft.com/it/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16  

## General structure

For the execution we use the following structure.

```
project
│   README.md
│   requirements.txt    
│
└───codice
│   └───main.py
│   └───preprocessing
│   │   │   preprocessing_train.py
│   │   │   preprocessing_test.py
│   │   │   __init__.py
│   └───models
│   │   │   attention.py
│   │   │   computation.py
│   │   │   transformers.py
│   │   │   __init__.py
│   │   └───bidirectional
│   │   │    └───10epochs
│   │   │        │  training_bidirectional_10.log
│   │   │        │  results_predictions_bi_10.csv
│   │   │    └───20epochs
│   │   │        │  training_bidirectional_20.log
│   │   │        │  results_predictions_bi_20.csv
│   │   └───monodirectional
│   │   │    └───10epochs
│   │   │        │  training_monodirectional_10.log
│   │   │        │  results_predictions_bi_10.csv
│   │   │    └───20epochs
│   │   │        │  training_monodirectional_20.log
│   │   │        │  results_predictions_bi_20.csv
└───dati
│   └───cnn
│       └───stories
│           └─ story_1
│           └─ story_2
│           └─ ...
└───final_data
│   │   x_tr.npy
│   │   x_val.npy
│   │   x_test.npy
│   │   y_tr.npy
│   │   y_val.npy
│   │   y_test.npy
└───tokenizers_vars
│   │   x_tokenizer.pickle
│   │   y_tokenizer.pickle
│   │   vars.pkl
└───data_transformer
│   │   results_predictions_transforme.csv
│   │   test_dataset.csv

```

## Execution

In order to execute all the files you must run:

`python main.py`

There are some paramaters that can be specified on the command line in order to change the hyperparameters without opening the code. In particular:

```
'-m', '--model', type=str, required=False, default="bidirectional_model",
    help="Insert a tipe of model between 'bidirectional' or 'monodirectional'!"
```

An example of a correct execution of the code is:

`python main.py -m "bidirectional_model"`

The project structure was created in order to make the code easily executable. In particular, the results of the model runs are already left in the directories. In fact, the algorithm has been built in such a way that, if it finds the results of the models inside the directories, it skips the execution part and goes directly to the evaluation part of the results. 