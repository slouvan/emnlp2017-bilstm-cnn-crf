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
import pickle
from util.preprocessing import perpareDataset, loadDatasetPickle, readCoNLL, remove_pkl_files,prepare_training_data, read_dict, get_target_task, build_vocab_from_domains, build_all_domain_term_dist, get_most_similar_domain, read_dict_data, set_target_task, compute_label_embeddings, get_nearest_labels
from keras import backend as K
from util.constants import  DATA_DIR, TASKS, NERS, NER_TAGS


parser = argparse.ArgumentParser(description="Experiment Slot Filling")
parser.add_argument("-target", dest="target_task", required=True, help="Target Task")
parser.add_argument("-strategy", dest="strategy", help="Strategy for resource selection", required=True, type=str)
parser.add_argument("-n", "--nb-sentence", dest="nb_sentence", help="Number of training sentence", type=int)
parser.add_argument("-ner", dest="ner", default=0, type=int)
parser.add_argument("-ner-name", dest="ner_name", default=None)
parser.add_argument("-diff-level", dest="different_level", default=0, type=int)
parser.add_argument("-ro", "--root-result", dest="root_dir_result", help="Root directory for results", default="results", type=str)
parser.add_argument("-label-embedding", dest="label_embedding", help="Label Embedding Cache", default=None, type=str)
parser.add_argument("-d", "--directory-name", dest="directory_name", help="Directory Name", required = True, type=str)
parser.add_argument("-r", "--run", dest="nb_run", default =1, type = int)
parser.add_argument("-p", "--param", dest="param_conf", help="Hyperparameters of the network", required=True, type=str)
parser.add_argument("-e", "--epoch", dest="nb_epoch", help="Number of epoch", default=50, type=int)
parser.add_argument("-t", "--tune", dest="tune", default=0, type=int)
parser.add_argument("-br", "--batch-range", dest="batch_range", default=None, type=str)
parser.add_argument("--filter-tags", dest="filter_tags", default=None, nargs="+", choices=NER_TAGS)
'''
parser.add_argument("-i", "--input", dest="input_dataset_conf", help="Input dataset configuration", required = True, type=str)
'''
args = parser.parse_args()


# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
if os.path.exists("/".join([args.root_dir_result,args.directory_name])) :
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


aux_task = []
target_task = [args.target_task]
if args.strategy == "most_similar" :
    word2idx = build_vocab_from_domains(TASKS)
    domain2term = build_all_domain_term_dist(TASKS, word2idx)
    most_similar_domain, score = get_most_similar_domain(args.target_task, TASKS, domain2term)
    print("Most similar domain is : {}".format(most_similar_domain))
    aux_task.append(most_similar_domain)
elif args.strategy == "all" :
    aux_task = list( set(TASKS) - set([args.target_task]))
    print(aux_task)

# Some network hyperparameters
params = read_dict(args.param_conf)
print("{} {}".format(type(params), params))

if args.ner == 1:
    if args.ner_name is None:
        aux_task = aux_task + NERS
        if args.different_level == 1:
            custom_classifier = {} # Assuming NER always on the bottom
            custom_classifier[target_task[0]] = [('LSTM', 100), 'CRF']
            for task in aux_task:
                if task in NERS :
                    custom_classifier[task] = ['CRF']
                else :
                    custom_classifier[task] = [('LSTM', 100), 'CRF']

            params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN',
                      'customClassifier': custom_classifier}
        else :
            params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN'}
    else :
        for NER in NERS:
            print(NER)
            if NER == args.ner_name:
                print("{} is the NER aux task".format(NER))
                aux_task.append(NER)
                break
        if args.different_level == 1:
            custom_classifier = {}  # Assuming NER always on the bottom
            custom_classifier[target_task[0]] = [('LSTM', 100), 'CRF']
            for task in aux_task:
                if task in NERS:
                    custom_classifier[task] = ['CRF']
                else:
                    custom_classifier[task] = [('LSTM', 100), 'CRF']

            params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN',
                      'customClassifier': custom_classifier}
        else :
            params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN'}

print("Target task : {}\t Aux task : {}".format(str(target_task), str(aux_task)))

datasets = read_dict_data(target_task + aux_task)
set_target_task(datasets, args.target_task)

# get the key where the dataset is the target task
if args.nb_sentence is not None :
    datasets[args.target_task]['nb_sentence'] = args.nb_sentence

relevant_label_embeddings = None
if args.label_embedding is not None:
    matrix = pickle.load(open(args.label_embedding+".emb", "rb"))
    idxs = pickle.load(open(args.label_embedding+".idxs", "rb"))
    relevant_label_embeddings = get_nearest_labels(args.target_task, aux_task, matrix, idxs)
    print(relevant_label_embeddings)

#remove_pkl_files()
#prepare_training_data(datasets, filter_tags=relevant_label_embeddings)
prepare_training_data(datasets, args)

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



if args.tune == 0:
    if args.nb_run == 1:
        model = BiLSTM(params)
        if args.batch_range is not None:
            model.setBatchRangeLength(args.batch_range)
        model.setMappings(mappings, embeddings)
        model.setDataset(datasets, data, mainModelName=args.target_task)  # KHUSUS MULTITSAK

        model.storeResults("/".join([args.root_dir_result,args.directory_name,"performance.out"])) #Path to store performance scores for dev / test
        model.predictionSavePath = "/".join([args.root_dir_result, args.directory_name,"predictions","[ModelName]_[Data].conll"]) #Path to store predictions
        model.modelSavePath = "/".join([args.root_dir_result,args.directory_name,"models/[ModelName].h5"]) #Path to store models

        model.fit(epochs=args.nb_epoch)
        model.saveParams("/".join([args.root_dir_result,args.directory_name,"param"]))
    else :

        for current_run in range(1, args.nb_run + 1):
            model = BiLSTM(params)
            if args.batch_range is not None:
                model.setBatchRangeLength(args.batch_range)
            model.setMappings(mappings, embeddings)
            model.setDataset(datasets, data, mainModelName=args.target_task)  # KHUSUS MULTITSAK

            model.storeResults("/".join([args.root_dir_result, args.directory_name + "_"+str(current_run), "performance.out"]))  # Path to store performance scores for dev / test
            model.predictionSavePath = "/".join([args.root_dir_result, args.directory_name + "_"+str(current_run), "predictions", "[ModelName]_[Data].conll"])  # Path to store predictions
            model.modelSavePath = "/".join([args.root_dir_result, args.directory_name + "_"+str(current_run), "models/[ModelName].h5"])  # Path to store models

            model.fit(epochs=args.nb_epoch)
            model.saveParams("/".join([args.root_dir_result, args.directory_name+ "_"+str(current_run), "param"]))

else :
    print("Tuning")
    drop_out_tuning = [0.25, 0.35, 0.45, 0.5]
    for current_drop_out in drop_out_tuning :
        params['dropout'] = (current_drop_out, current_drop_out)
        model = BiLSTM(params)
        if args.batch_range is not None:
            model.setBatchRangeLength(args.batch_range)
        model.setMappings(mappings, embeddings)
        model.setDataset(datasets, data, mainModelName=args.target_task)  # KHUSUS MULTITSAK

        model.storeResults("/".join([args.root_dir_result, args.directory_name, "performance.out"]))  # Path to store performance scores for dev / test
        model.predictionSavePath = "/".join([args.root_dir_result, args.directory_name, "predictions", "[ModelName]_[Data].conll"])  # Path to store predictions
        model.modelSavePath = "/".join([args.root_dir_result, args.directory_name, "models/[ModelName].h5"])  # Path to store models
        model.fit(epochs=args.nb_epoch)
        model.saveParams("/".join([args.root_dir_result, args.directory_name, "param"]))
        model.saveParamTuningResults("/".join([args.root_dir_result, args.directory_name, "tuning_results"]))


