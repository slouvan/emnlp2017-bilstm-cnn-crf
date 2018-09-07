import os


dirname = "/Users/slouvan/sandbox/emnlp2017-bilstm-cnn-crf/result_analyze/"
base_exp_dir = "MTL_MIT_Movie_CONLL_NB_SENT_"

for i in range(10,110, 10):
    path = os.path.join(dirname, base_exp_dir+str(i))
    #print("Path = {}".format(path))
    with open(os.path.join(path, "performance.out"), "r") as f:
        lines = f.readlines()
        last_line = lines[len(lines) - 1]
        field = last_line.split("\t")
        print(field[len(field) - 1], end="")
