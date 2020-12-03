import os
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.applications import Xception
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, GlobalMaxPooling2D

class networkClass:
    def __init__(self, param):
        self.network_patch_size = param.network_patch_size
        self.no_of_classes = len(param.all_class_names)
        self.checkpoint_dir = param.checkpoint_dir
        self.lr = param.learning_rate

    def get_model(self):
        # input_tensor = Input(shape=(self.network_patch_size, self.network_patch_size, 3))
        input_tensor = Input(shape=(None, None, 3))
        baseModel = Xception(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=None,
                            pooling=None)

        x = baseModel(input_tensor)
        x = GlobalMaxPooling2D()(x)

        predictions = Dense(self.no_of_classes, activation='softmax', name='classifier')(x)
        final_model = Model(inputs=input_tensor , outputs=predictions, name='cyto_xception')

        # final_model.summary()
        return final_model

    def optimize(self, model):
        opt = Adam(self.lr)
        model.compile(optimizer=opt, loss=self.focal_loss(), metrics=['accuracy'])

    def focal_loss(self, gamma=2., alpha=.25):
        def categorical_focal_loss_fixed(y_true, y_pred):
            # Scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

            # Clip the prediction value to prevent NaN's and Inf's
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

            # Calculate Cross Entropy
            # cross_entropy = K.categorical_crossentropy(y_true, y_pred)
            cross_entropy = -y_true * K.log(y_pred)

            # Calculate Focal Loss
            loss = K.pow(1 - y_pred, gamma) * cross_entropy
            # loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

            # Sum the losses in mini_batch
            return K.sum(loss, axis=1)

        return categorical_focal_loss_fixed

    def load_checkpoint(self, model):
        if os.path.isfile(self.checkpoint_dir):
            print("Resumed model's weights from {}".format(self.checkpoint_dir))
            model.load_weights(self.checkpoint_dir)
        else:
            print('No checkpoint found!')
            exit()