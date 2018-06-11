#!/bin/sh
now=$(date +"%T")
echo "Current time : $now"
python Train_ATIS.py -l 0.01
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_MultiTask.py -l 0.01
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_ATIS.py -l 0.05
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_MultiTask.py -l 0.05
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_ATIS.py -l 0.1
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_MultiTask.py -l 0.1
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_ATIS.py -l 0.2
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_MultiTask.py -l 0.2
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_ATIS.py -l 0.4
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_MultiTask.py -l 0.4
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_ATIS.py -l 0.6
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_MultiTask.py -l 0.6
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_ATIS.py -l 0.8
wait
now=$(date +"%T")
echo "Current time : $now"
python Train_MultiTask.py -l 0.8
wait
#python Train_ATIS.py -l 0.4
#wait
#python Train_ATIS.py -l 0.6
#wait
#python Train_ATIS.py -l 0.8
#wait
#python Train_ATIS.py -l 1.0
#wait
