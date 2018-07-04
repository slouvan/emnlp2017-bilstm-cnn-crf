# This script trains the BiLSTM-CRF architecture for part-of-speech tagging using
# the universal dependency dataset (http://universaldependencies.org/).
# The code use the embeddings by Komninos et al. (https://www.cs.york.ac.uk/nlp/extvec/)
from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
import argparse
from util.preprocessing import perpareDataset, loadDatasetPickle, prepare_training_data, read_dict


# :: Parse arguments
parser = argparse.ArgumentParser(description="Experiment Slot Filling")
parser.add_argument("-n", "--nb-sentence", dest="nb_sentence", help="Number of training sentence", type=int)
parser.add_argument("-d", "--directory-name", dest="directory_name", help="Directory Name", required = True, type=str)
parser.add_argument("-i", "--input", dest="input_dataset_conf", help="Input dataset configuration", required = True, type=str)
parser.add_argument("-p", "--param", dest="param_conf", help="Hyperparameters of the network", required=True, type=str)
parser.add_argument("-e", "--epoch", dest="nb_epoch", help="Number of epoch", default=50, type=int)
args = parser.parse_args()

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# The  directory to store results
if os.path.exists("/".join(["results",args.directory_name])) :
    raise ValueError("The directory {} exists".format(args.directory_name))
else :
    print("The directory does not exist")

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

datasets = read_dict(args.input_dataset_conf)
print("{} {}".format(type(datasets), datasets))

# :: Needed for simulating the low resource scenarios
if args.nb_sentence is not None :
    datasets[list(datasets.keys())[0]]['nb_sentence'] = args.nb_sentence

prepare_training_data(datasets)

# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
embeddingsPath = 'komninos_english_embeddings.gz'

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
params = read_dict(args.param_conf)

model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults("/".join(["results",args.directory_name,"performance.out"])) #Path to store performance scores for dev / test
model.predictionSavePath = "/".join(["results", args.directory_name,"predictions","[ModelName]_[Epoch]_[Data].conll"]) #Path to store predictions
model.modelSavePath = "/".join(["results",args.directory_name,"models/model_[DevScore]_[TestScore]_[Epoch].h5"]) #Path to store models
model.fit(epochs=args.nb_epoch)