
import sys, os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import model_from_json

from datetime import datetime
#Local methods:
from data_loader import data_loader
# import cv2
sys.path.append('../')
# from generator import Generator

import pdb
class Model :
    def __init__(self, mode='test') :
        self.mode = mode
        self.input_image_size = (227, 227)
        self.label_indices = {'road_bikes': 0,'mountain_bikes':1}
        self.labels_ordered = list(self.label_indices)
        self.num_classes = len(self.label_indices)
        # self.save_dir = "tf_data/sample_model"
        self.train_test_split=0.8

        self.channel_means = np.array([147.12697, 160.21092, 167.70029])

    def train(self):
        data = data_loader(label_indices = self.label_indices, 
           channel_means = self.channel_means, 
           train_test_split = self.train_test_split,
           input_image_size = self.input_image_size, 
           data_path = 'data')
        # glob.glob('data/*/*.jpg')
        input_shape = (227, 227,3)
        model = Sequential()
        # pdb.set_trace()
        batch_size = 5
        epochs = 500
        model.add(Conv2D(64, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(16, kernel_size=(9, 9),
                 activation='relu',
                 input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(5, 5)))
        model.add(Dropout(0.25))

        # model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        train_datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True,   # randomly flip images
            rescale = 1./255,
        )
        test_datagen = ImageDataGenerator(
            rescale = 1./255,
            horizontal_flip = True,
            fill_mode = "nearest",
            zoom_range = 0.0,
            width_shift_range = 0.3,
            height_shift_range=0.3,
            rotation_range=30)

        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
        model.fit_generator(
            train_datagen.flow(
                data.train.X, data.train.y, batch_size=batch_size),    
            validation_data = test_datagen.flow(data.test.X,data.test.y,batch_size=batch_size),
            validation_steps=len(data.test.X) / batch_size,steps_per_epoch=len(data.train.X) / batch_size, 
            epochs=epochs,callbacks=[TrainValTensorBoard(write_graph=False)])

        # model.fit(data.train.X, data.train.y,
        #           batch_size=batch_size,
        #           epochs=epochs,
        #           verbose=1,
        #           validation_data=(data.test.X, data.test.y))
        score = model.evaluate(data.test.X, data.test.y, verbose=0)
        model_json = model.to_json()
        with open("model.json","w") as json_file:
             json_file.write(model_json)

        model.save('weights.h5')

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def load_previous_model(self):
        model_json_file="model.json"
        model_weights_file= "weights.h5"

        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()


    def test(self,data):
        self.load_previous_model()
        self.preds = self.loaded_model.predict(data.test.X)
        # pdb.set_trace()
        temp = sum(np.argmax(data.test.y,1) == np.argmax(self.preds,1))
        score = temp/len(data.test.y)
        return score

    def predict(self,data) :
        self.load_previous_model()
        self.label_indices = {'road_bikes': 0,'mountain_bikes':1}
        predicted_value = np.argmax(self.loaded_model.predict(data),1)
        # pdb.set_trace()
        for (key,value) in self.label_indices.items():
            if value == predicted_value[0] :
                return key


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
