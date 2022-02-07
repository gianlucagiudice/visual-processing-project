from datetime import datetime

import keras.engine.functional
import numpy as np

from src.models.Model import Model as MyModel

import tensorflow as tf
from src.config import OUTPUT_IMAGE_FOLDER, OUTPUT_REPORT_FOLDER, CHECKPOINT_DIR, LOG_DIR

import keras

from os.path import join
import os
import re
from keras.layers import (Dropout, BatchNormalization, Activation)

from keras.layers import Dense
from keras.models import Model
import pickle

import visualkeras

import sys

LR = 0.001

EPOCHS = 50
BATCH_SIZE = 512
PATIENCE = 5
DROPOUT = 0.5
LAST_LAYER = 'flatten'

def print_tensorboard_command():
    path = os.path.normpath(PretrainedVGG.log_dir)
    path = os.path.join(*path.split(os.sep)[1:])
    print(f'Tensorboard: tensorboard --logdir {path}')


class PretrainedVGG(MyModel):
    IMAGE_INPUT_SIZE = (224, 224, 3)
    
    checkpoint_dir = join(CHECKPOINT_DIR, 'pretrained_vgg')
    checkpoint_filepath = join(checkpoint_dir, 'ckpt-{epoch:03d}.hdf5')
    log_dir = join(LOG_DIR, 'fit', 'pretrained_vgg/') + datetime.now().strftime("%Y%m%d-%H%M%S")

    def __init__(self, last_layer=LAST_LAYER, input_size=IMAGE_INPUT_SIZE):
        super().__init__(input_size)
        # Backbone: VGG16
        self.model = self.init_model(last_layer)
        # Reset weights
        #self.model = self.reset_weights(self.model)
        # Add output layers
        self.model = self.add_output_layers(self.model)
        # Save network architecture
        self.save_plot_network()
        # Save summary
        self.save_summary_output()

    def init_model(self, last_layer):
        base_vgg16_model = tf.keras.applications.VGG16(include_top=True, pooling='avg', weights="imagenet")
        vgg16_model = Model(inputs=base_vgg16_model.input, outputs=base_vgg16_model.get_layer(last_layer).output)
        # Blocking the weights of the previous layers
        for layer in vgg16_model.layers:
            if layer.name == 'block4_conv1':
                break
            layer.trainable = False

        for layer in vgg16_model.layers:
            print(layer.name + ' = ', layer.trainable)

        return vgg16_model


    def train(self, x_train, y_train, x_val, y_val, x_test, y_test) -> None:
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Define callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            mode='min',
            monitor='val_loss',
            patience=PATIENCE
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1, update_freq='batch', profile_batch=0)

        # Fit model
        print_tensorboard_command()

        print('>>> Start training')
        history = self.model.fit(x=x_train,
                                 y={'gender_output': y_train['gender'], 'age_output': y_train['age']},
                                 validation_data=(x_val, [y_val['gender'], y_val['age']]),
                                 use_multiprocessing=True,
                                 workers=os.cpu_count(),
                                 callbacks=[early_stopping_callback, model_checkpoint_callback, tensorboard_callback],
                                 epochs=EPOCHS)

        # Dump history dictinary
        with open(join(self.checkpoint_dir, 'pretrained_vgg.pickle'), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Load best weights
        #self.load_best_weights()

        # Evaluate model
        self.evaluate(x_test, y_test)


    def load_best_weights(self):
        ckpt_list = [(x, re.findall('-(\d+)', x)[0]) for x in os.listdir(self.checkpoint_dir) if re.match('ckpt-', x)]
        best_model = sorted(ckpt_list, key=lambda x: x[1], reverse=True)[0][0]
        self.model.load_weights(join(self.checkpoint_dir, best_model))

    def evaluate(self, x_test, y_test):
        results = self.model.evaluate(x=x_test,
                                      y={'gender_output': y_test['gender'], 'age_output': y_test['age']})
        # Dump evaluation result
        with open(join(self.checkpoint_dir, 'pretrained_vgg.pickle'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_plot_network(self):
        file_path = join(OUTPUT_IMAGE_FOLDER, 'pretrained_vgg.png')
        visualkeras.layered_view(self.model, legend=True, to_file=file_path,
                                 scale_xy=2, scale_z=2, max_z=6, spacing=5)

    def save_summary_output(self):
        file_path = join(OUTPUT_REPORT_FOLDER, 'pretrained_vgg.txt')
        with open(file_path, 'w') as f:
            orig_stdout = sys.stdout
            sys.stdout = f
            print(self.model.summary())
            sys.stdout = orig_stdout

    def save_weights(self) -> None:
        pass

    def load_weights(self):
        pass

    '''
    @staticmethod
    def reset_weights(model):
        for i, layer in enumerate(model.layers):
            if hasattr(model.layers[i], 'kernel_initializer') and hasattr(model.layers[i], 'bias_initializer'):
                weight_initializer = model.layers[i].kernel_initializer
                bias_initializer = model.layers[i].bias_initializer

                old_weights, old_biases = model.layers[i].get_weights()

                model.layers[i].set_weights([
                    weight_initializer(shape=old_weights.shape),
                    bias_initializer(shape=old_biases.shape)
                ])
        return model
    '''

    def predict(self, image: np.array) -> (bool, int):
        pass

    @staticmethod
    def add_output_layers(starting_model: keras.engine.functional.Functional):
        # Since in the VGG constructor I specified the "include_top = False", the last 2 layers of the network
        # are removed. By including "pooling = avg" the network is constructed inserting a GlobalAveragePooling2D
        # 2 new layers

        # Output layer of the model
        final_layer = starting_model.output




        # Gender layer
        gender_layer = Dropout(DROPOUT)(Activation('relu')(Dense(64)(Dropout(DROPOUT)(final_layer))))
        gender_layer = Dense(1, name = 'gender_output', activation = 'sigmoid')(BatchNormalization()(gender_layer))

        # Age layer
        age_layer = Dropout(DROPOUT)(Activation('relu')(Dense(64)(Dropout(DROPOUT)(final_layer))))
        age_layer = Dense(1, name='age_output', activation='sigmoid')(BatchNormalization()(age_layer))

        # Final model
        final_model = Model(inputs=starting_model.input, outputs=[gender_layer, age_layer],
                            name='pretrained_vgg_age_gender_clf')
        # Compile the model
        losses = {
            "gender_output": "binary_crossentropy",
            "age_output": "mean_squared_error",
        }
        #lossWeights = {
        #    "gender_output": 3.0,
        #    "age_output": 1.0
        #}
        metrics = {
            "gender_output": 'accuracy',
            "age_output": 'mean_absolute_error'
        }
        optimizer = tf.keras.optimizers.Adam(LR)

        final_model.compile(loss=losses,
                            #loss_weights=lossWeights,
                            metrics=metrics, optimizer=optimizer)

        return final_model












