#!/bin/bash

python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 200 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_200_CONLL_2003_NER_Filtered_LOC -p params/MTL_Default_Param --batch-range max --filter-tags LOC
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 400 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_400_CONLL_2003_NER_Filtered_LOC -p params/MTL_Default_Param --batch-range max --filter-tags LOC
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Restaurant -strategy none -n 800 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Restaurant_800_CONLL_2003_NER_Filtered_LOC -p params/MTL_Default_Param --batch-range max --filter-tags LOC
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 200 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_200_CONLL_2003_NER_Filtered_PER -p params/MTL_Default_Param --batch-range max --filter-tags PER
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 400 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_400_CONLL_2003_NER_Filtered_PER -p params/MTL_Default_Param --batch-range max --filter-tags PER
wait
python Train_MTL_Sequence_Tagging_Selective.py -target MIT_Movie -strategy none -n 800 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_MIT_Movie_800_CONLL_2003_NER_Filtered_PER -p params/MTL_Default_Param --batch-range max --filter-tags PER
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 200 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_200_CONLL_2003_NER_Filtered_LOC -p params/MTL_Default_Param --batch-range max --filter-tags LOC
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 400 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_400_CONLL_2003_NER_Filtered_LOC -p params/MTL_Default_Param --batch-range max --filter-tags LOC
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 800 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_800_CONLL_2003_NER_Filtered_LOC -p params/MTL_Default_Param --batch-range max --filter-tags LOC
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 200 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_200_CONLL_2003_NER_Filtered_ORG -p params/MTL_Default_Param --batch-range max --filter-tags ORG
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 400 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_400_CONLL_2003_NER_Filtered_ORG -p params/MTL_Default_Param --batch-range max --filter-tags ORG
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 800 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_800_CONLL_2003_NER_Filtered_ORG -p params/MTL_Default_Param --batch-range max --filter-tags ORG
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 200 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_200_CONLL_2003_NER_Filtered_LOC_ORG -p params/MTL_Default_Param --batch-range max --filter-tags LOC ORG
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 400 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_400_CONLL_2003_NER_Filtered_LOC_ORG -p params/MTL_Default_Param --batch-range max --filter-tags LOC ORG
wait
python Train_MTL_Sequence_Tagging_Selective.py -target ATIS -strategy none -n 800 -ner 1 -ner-name CONLL_2003_NER -ro result_analyze -d MTL_ATIS_800_CONLL_2003_NER_Filtered_LOC_ORG -p params/MTL_Default_Param --batch-range max --filter-tags LOC ORG