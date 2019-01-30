import numpy as np
import time
from sample_model import Model
from data_loader import data_loader
# from generator import Generator
from optparse import OptionParser
# import pdb
import cv2

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
   
def evaluate_with_path(image_path) :
    im = cv2.imread(image_path)
    input_image_size = (227, 227)
    im = cv2.resize(im, (input_image_size[0], input_image_size[1]))
    channel_means = np.array([147.12697, 160.21092, 167.70029])
    X = np.zeros((1, input_image_size[0], input_image_size[1], 3), 
                            dtype = 'float32')

    X[0,:, :, :] = im - channel_means

    M = Model(mode = 'test')

    #Evaluate on test images:
    # print(f'Predicted {}')
    return M.predict(X)
    

if __name__ == '__main__':
    # def parse_cmd_opts():

    #     parser = argparse.ArgumentParser(description='Add jpg file to process.')
    #     parser.add_argument('--bike_image', '-bk', dest='input_image',
    #                         nargs=1, metavar='jpg', required=True,
    #                      help='Input bike image')
    #     args = parser.parse_args()
    #     return args
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")
    parser.add_option("-i", "--image",
                      action="store_true",
                      dest="image",
                      default=False,
                      help="Add an image path")
    (options, args) = parser.parse_args()

    if len(args) == 1:
        # pdb.set_trace()
        predicted = evaluate_with_path(args[0])
        # cv2.namedWindow(predicted, cv2.WINDOW_NORMAL)
        cv2.imshow(predicted,cv2.imread(args[0]))

        # cv2.imshow(args[0],predicted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else : 
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
