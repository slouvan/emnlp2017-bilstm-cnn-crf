# This file contain an example how to perform multi-task learning using the
# BiLSTM-CNN-CRF implementation.
# In the datasets variable, we specify two datasets: POS-tagging (unidep_pos) and conll2000_chunking.
# The network will then train jointly on both datasets.
# The network can on more datasets by adding more entries to the datasets dictionary.

from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
import argparse
from util.preprocessing import perpareDataset, loadDatasetPickle, readCoNLL, remove_pkl_files,prepare_training_data
from keras import backend as K

parser = argparse.ArgumentParser(description="Experiment Slot Filling")
parser.add_argument("-l", "--labeling-rate", dest="labeling_rate", help="Labeling Rate", metavar="N", type=float)
args = parser.parse_args()

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
#datasets = {
#    'unidep_pos':
#        {'columns': {1:'tokens', 3:'POS'},
#         'label': 'POS',
#         'evaluate': True,
#         'commentSymbol': None},
#    'conll2000_chunking':
#        {'columns': {0:'tokens', 2:'chunk_BIO'},
#         'label': 'chunk_BIO',
#         'evaluate': True,
#         'commentSymbol': None},
#}

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
         'proportion': 0.6,
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


labeling_rate = 0.0
if args.labeling_rate is not None :
    datasets['MIT_Restaurant']['proportion'] = args.labeling_rate
else :
    datasets['MIT_Restaurant']['proportion'] = 1

print("Labeling rate is set to : {} ".format(datasets['MIT_Restaurant']['proportion']))


#remove_pkl_files()
prepare_training_data(datasets)

embeddingsPath = 'komninos_english_embeddings.gz' #Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets, reducePretrainedEmbeddings=True)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN'}


model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data, mainModelName='MIT_Restaurant')  # KHUSUS MULTITSAK

model.storeResults('results/MIT_Restaurant_CONLL_Multitask_'+str(datasets['MIT_Restaurant']['proportion'])+'.csv') #Path to store performance scores for dev / test
model.predictionSavePath = "results/[ModelName]_MultiTask_"+str(datasets['MIT_Restaurant']['proportion'])+"_[Epoch]_[Data].conll" #Path to store predictions
model.modelSavePath = "models/[ModelName]_Multitask_"+str(datasets['MIT_Restaurant']['proportion'])+"_[[DevScore]_[TestScore]_[Epoch].h5" # labeling_rate
model.customizedAlternate = True      # Additional params
model.fit(epochs=50)



