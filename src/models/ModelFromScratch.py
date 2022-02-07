import os
import pickle
import re
import shutil
import sys
from datetime import datetime
from os.path import join

import keras.engine.functional
import numpy as np
import tensorflow as tf
import visualkeras
from keras.layers import Dense, MaxPool2D, Conv2D, GlobalAveragePooling2D
from keras.layers import (Dropout, BatchNormalization, Activation)
from keras.models import Model
import keras_tuner as kt

from src.config import OUTPUT_IMAGE_FOLDER, OUTPUT_REPORT_FOLDER, CHECKPOINT_DIR, LOG_DIR, SAVE_MODEL_DIR
from src.models.Model import Model as MyModel

LR = 0.001

EPOCHS = 25
BATCH_SIZE = 512
PATIENCE = 5
DROPOUT = 0.4


def print_tensorboard_command():
    path = os.path.normpath(ModelFromScratch.log_dir_training)
    path = os.path.join(*path.split(os.sep)[1:])
    print(f'Tensorboard: tensorboard --logdir {path}')


def init_temp_dirs():
    # Remove directories
    shutil.rmtree(ModelFromScratch.log_dir, ignore_errors=True)
    shutil.rmtree(ModelFromScratch.checkpoint_dir, ignore_errors=True)
    # Create directories
    os.makedirs(ModelFromScratch.log_dir, exist_ok=True)
    os.makedirs(ModelFromScratch.checkpoint_dir, exist_ok=True)


class ModelFromScratch(MyModel):
    IMAGE_INPUT_SIZE = (124, 124, 3)
    
    checkpoint_dir = join(CHECKPOINT_DIR, 'from_scratch')
    checkpoint_filepath = join(checkpoint_dir, 'ckpt-{epoch:03d}.h5')
    log_dir = join(LOG_DIR, 'fit', 'from_scratch/')
    log_dir_training = log_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = 'from_scratch'

    # Define callbacks
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        mode='min',
        monitor='val_loss',
        patience=PATIENCE
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    def __init__(self, input_size=IMAGE_INPUT_SIZE):
        super().__init__(input_size)
        self.input_size = input_size
        # Backbone: MobileNetV3Small
        self.model = self.init_model(self.input_size)
        # Add output layers
        self.model = self.add_output_layers(self.model)
        # Save network architecture
        self.save_plot_network()
        # Save summary
        self.save_summary_output()

    def build_model(self, hp):
        dropout = hp.Float(
            'dropout_rate',
            min_value=0.1,
            max_value=0.5,
            default=0.4,
            step=0.05,
        )

        lr = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])

        model = self.init_model(self.input_size)
        model = self.add_output_layers(model, lr=lr, dropout=dropout)
        return model

    def hyperparameter_optimization(self, X_train, y_train, X_val, y_val, X_test, y_test, max_trials=20):
        log_dir_tuner = LOG_DIR + '_tuner_search'
        tuner = kt.BayesianOptimization(
            self.build_model,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=1,
            directory=log_dir_tuner
            )
        tuner.search(X_train, {'gender_output': y_train['gender'], 'age_output': y_train['age']},
                     validation_data=(X_val, [y_val['gender'], y_val['age']]),
                     callbacks=[ModelFromScratch.early_stopping_callback,
                                ModelFromScratch.model_checkpoint_callback],
                     epochs=EPOCHS
                     )
        # Dump tuner info
        with open(f"{log_dir_tuner}/tuner.pkl", "wb") as f:
            pickle.dump(tuner, f)
        # Save tuner info
        original_stdout = sys.stdout
        with open(f"{log_dir_tuner}/tuner.txt", "w") as f:
            sys.stdout = f
            print(tuner.search_space_summary())
            print(f'\n{"-"*100}\n')
            print(tuner.results_summary())
            sys.stdout = original_stdout
        # Print tuner info
        print(tuner.search_space_summary())
        print(tuner.results_summary())
        # Save model
        self.model = tuner.get_best_models()[0]
        # Save best weights
        self.save_weights()
        # Evaluate model
        self.evaluate(X_test, y_test)

    def predict(self, image: np.array) -> (bool, int):
        return self.model.predict(image)

    def extract_feature(self, image):
        output_1 = 'activation_features_1'
        output_2 = 'activation_features_2'

        intermediate_1 = keras.models.Model(inputs=self.model.layers[0].input,
                                            outputs=self.model.get_layer(output_1).output)
        intermediate_2 = keras.models.Model(inputs=self.model.layers[0].input,
                                            outputs=self.model.get_layer(output_2).output)

        features_1 = intermediate_1.predict(image)
        features_2 = intermediate_2.predict(image)

        return np.concatenate([features_1, features_2], axis=1).flatten()

    def get_input_shape(self):
        return self.model.layers[0].input.shape[1:]

    def train(self, x_train, y_train, x_val, y_val, x_test, y_test) -> None:
        # Init temp directories
        init_temp_dirs()

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir_training, histogram_freq=1, update_freq='batch')

        # Fit model
        print(self.model.summary())
        print_tensorboard_command()

        print('>>> Start training')
        history = self.model.fit(x=x_train,
                                 y={'gender_output': y_train['gender'], 'age_output': y_train['age']},
                                 validation_data=(x_val, [y_val['gender'], y_val['age']]),
                                 use_multiprocessing=True,
                                 workers=os.cpu_count(),
                                 callbacks=[ModelFromScratch.early_stopping_callback,
                                            ModelFromScratch.model_checkpoint_callback, tensorboard_callback],
                                 epochs=EPOCHS)

        # Dump history dictinary
        with open(join(self.checkpoint_dir, 'from_scratch_training_history.pickle'), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Load best weights
        self.load_best_weights()

        # Save the model
        self.save_weights()

        # Evaluate model
        self.evaluate(x_test, y_test)

    def load_best_weights(self):
        ckpt_list = [(x, re.findall('-(\d+)', x)[0]) for x in os.listdir(self.checkpoint_dir) if re.match('ckpt-', x)]
        best_model = sorted(ckpt_list, key=lambda x: x[1], reverse=True)[0][0]
        self.model.load_weights(join(self.checkpoint_dir, best_model))

    def evaluate(self, x_test, y_test, path=None, dump=True):
        results = self.model.evaluate(x=x_test,
                                      y={'gender_output': y_test['gender'], 'age_output': y_test['age']})
        # Dump evaluation result
        if path is None:
            path = join(self.checkpoint_dir, 'from_scratch_evaluation.pickle')
        if dump:
            with open(path, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return results

    def save_plot_network(self):
        file_path = join(OUTPUT_IMAGE_FOLDER, 'model_from_scratch_plot.png')
        visualkeras.layered_view(self.model, legend=True, to_file=file_path,
                                 scale_xy=2, scale_z=2, max_z=6, spacing=5)

    def save_summary_output(self):
        file_path = join(OUTPUT_REPORT_FOLDER, 'model_from_scratch_summary.txt')
        with open(file_path, 'w') as f:
            orig_stdout = sys.stdout
            sys.stdout = f
            print(self.model.summary())
            sys.stdout = orig_stdout

    def save_weights(self) -> None:
        self.model.save(join(SAVE_MODEL_DIR, ModelFromScratch.model_name + '.h5'))

    def load_weights(self, path='../model/from_scratch_best.h5'):
        self.model = keras.models.load_model(path)

    @staticmethod
    def init_model(input_size):
        # Neural Net
        input = keras.layers.Input(shape=input_size)

        # Hidden convolutional layers
        h_conv1 = MaxPool2D((2, 2))(
            Activation('relu')(BatchNormalization(axis=3)(Conv2D(32, 3, padding='same')(input))))
        h_conv2 = MaxPool2D((2, 2))(
            Activation('relu')(BatchNormalization(axis=3)(Conv2D(64, 3, padding='same')(h_conv1))))
        h_conv3 = MaxPool2D((2, 2))(
            Activation('relu')(BatchNormalization(axis=3)(Conv2D(128, 3, padding='same')(h_conv2))))
        h_conv4 = MaxPool2D((2, 2))(
            Activation('relu')(BatchNormalization(axis=3)(Conv2D(256, 3, padding='same')(h_conv3))))
        h_conv5 = MaxPool2D((2, 2))(
            Activation('relu')(BatchNormalization(axis=3)(Conv2D(256, 3, padding='same')(h_conv4))))

        # Flatten layers after convolutions
        h_conv5_flat = GlobalAveragePooling2D()(h_conv5)

        return Model(inputs=input, outputs=h_conv5_flat)

    @staticmethod
    def add_output_layers(starting_model: keras.engine.functional.Functional,
                          lr=LR, dropout=DROPOUT):

        # Output layer of the model
        final_layer = starting_model.output

        # Gender layer
        gender_layer = \
            Dropout(rate=dropout)(
                Activation(name='activation_features_1', activation='relu')(
                    BatchNormalization()(
                        Dense(units=256)(
                            final_layer))))
        gender_layer = \
            Dropout(rate=dropout)(
                Activation('relu')(
                    BatchNormalization()(
                        Dense(units=128)(
                            gender_layer))))
        gender_layer = Dense(1, name='gender_output', activation='sigmoid')(gender_layer)

        # Age layer
        age_layer = \
            Dropout(rate=dropout)(
                Activation(name='activation_features_2', activation='relu')(
                    BatchNormalization()(
                        Dense(units=256)(
                            final_layer))))
        age_layer = \
            Dropout(rate=dropout)(
                Activation('relu')(
                    BatchNormalization()(
                        Dense(units=128)(
                            age_layer))))
        age_layer = Dense(1, name='age_output', activation='linear')(age_layer)

        # Final model
        final_model = Model(inputs=starting_model.input, outputs=[gender_layer, age_layer],
                            name='from_scratch_age_gender_clf')
        # Compile the model
        losses = {
            "gender_output": "binary_crossentropy",
            "age_output": "mean_squared_error",
        }
        lossWeights = {
            "gender_output": 3.0,
            "age_output": 1.0
        }
        metrics = {
            "gender_output": 'accuracy',
            "age_output": 'mean_absolute_error'
        }
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        final_model.compile(loss=losses, loss_weights=lossWeights, metrics=metrics, optimizer=optimizer)
        return final_model
