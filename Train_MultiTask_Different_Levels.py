# This file contain an example how to perform multi-task learning on different levels.
# In the datasets variable, we specify two datasets: POS-tagging (unidep_pos) and conll2000_chunking.
# We pass a special parameter to the network (customClassifier), that allows that task are supervised at different levels.
# For the POS task, we use one shared LSTM layer followed by a softmax classifier. However, the chunking
# task uses the shared LSTM layer, then a task specific LSTM layer with 50 recurrent units, and then a CRF classifier.

from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle,prepare_training_data

from keras import backend as K

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################
'''
datasets = {
    'unidep_pos':
        {'columns': {1:'tokens', 3:'POS'},
         'label': 'POS',
         'evaluate': True,
         'commentSymbol': None},
    'conll2000_chunking':
        {'columns': {0:'tokens', 2:'chunk_BIO'},
         'label': 'chunk_BIO',
         'evaluate': True,
         'commentSymbol': None},
}
'''

######################################################
#
# Data preprocessing
#
######################################################
datasets = {
    'MIT_Restaurant':                            #Name of the dataset
        {'columns': {0:'tokens', 1:'restaurant_BIO'},   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
         'label': 'restaurant_BIO',                     #Which column we like to predict
         'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None,
         'proportion': 1,
         'ori': True,
         'targetTask': True},
    'CONLL_2003_NER':                            #Name of the dataset
        {'columns': {0:'tokens', 1:'CONLL_2003_BIO'},   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
         'label': 'CONLL_2003_BIO',                     #Which column we like to predict
         'evaluate': False,                   #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None,
         'targetTask' : False,
         'proportion' : 1,
         'ori': True,},              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}

prepare_training_data(datasets)


embeddingsPath = 'komninos_english_embeddings.gz' #Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
#params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25),'charEmbeddings': 'CNN',
#          'customClassifier': {'unidep_pos': ['Softmax'], 'conll2000_chunking': [('LSTM', 50), 'CRF']}}
params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25),'charEmbeddings': 'CNN',
          'customClassifier': {'MIT_Restaurant': ['CRF'], 'CONLL_2003_NER': [('LSTM', 50), 'CRF']}}
model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults('results/MIT_Restaurant_CONLL_MultitaskDifferentLevel_'+str(datasets['MIT_Restaurant']['proportion'])+'.csv') #Path to store performance scores for dev / test

model.modelSavePath = "models/[ModelName]_MultitaskDifferentLevel_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=50)



