# Functions: Alphabet Classifier using NN

import numpy as np

def next_batch(batch_size, data, labels):
    idx = np.arange(0 , len(data)) # 0 ~ len(data)까지 1단위로 배열 생성
    np.random.shuffle(idx) # 생성된 배열 (idx)를 무작위로 섞음.
    idx = idx[:batch_size] # num (batch size) 까지만 한정.
    data_shuffle = [data[i] for i in idx] # 무작위로 섞인 입력 데이터
    labels_shuffle = [labels[i] for i in idx] # 무작위로 섞인 입력과 동일한 순서로 정렬된 출력 데이터
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")