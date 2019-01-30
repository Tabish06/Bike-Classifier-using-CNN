import numpy as np
import time
from sample_model import Model
from data_loader import data_loader
# from generator import Generator


def evaluate(label_indices = {'mountain_bikes': 1, 'road_bikes': 0},
             channel_means = np.array([147.12697, 160.21092, 167.70029]),
             data_path = 'test_data',
             minibatch_size = 32,
             num_batches_to_test = 10,
             checkpoint_dir = 'tf_data/sample_model'):

    
    print("1. Loading data")
    data = data_loader(label_indices = label_indices, 
               		   channel_means = channel_means,
               		   train_test_split = 0, 
               		   data_path = data_path)

    print("2. Instantiating the model")
    M = Model(mode = 'test')

    #Evaluate on test images:
    accuracy = M.test(data)
    return accuracy
   


if __name__ == '__main__':
    program_start = time.time()
    accuracy = evaluate()
    score = accuracy
    program_end = time.time()
    total_time = round(program_end - program_start,2)
    print()
    print("Execution time (seconds) = ", total_time)
    print('Accuracy = ' + str(accuracy))
    print("Score = ", score)
    print()
