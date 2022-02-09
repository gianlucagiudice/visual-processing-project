import os.path
from enum import Enum
from os import path
from os.path import join

import numpy as np
import pandas as pd
import telepot
import time
import cv2
from PIL import Image
from urllib.request import urlopen

import keras
import tensorflow.compat.v1 as tf

from src.EnhancementUtils import EnhancementUtils
from src.config import IMDB_CROPPED_PATH, IMDB_FAMOUS_ACTORS_FILENAME, WIKI_PATH, WIKI_FAMOUS_ACTORS_FILENAME
from src.detection.yolo.YoloFaceDetector import YoloFaceDetector
from src.detection.cascade.CascadeFaceDetector import CascadeFaceDetector
#from src.models.Model import IMAGE_INPUT_SIZE

from src.DataManager import DataManager
from src.retrieval.ImageSimilarity import ImageSimilarity

METADATA_IMDB_FILE = '../../dataset/imdb_crop/imdb_most_famous_actors.pickle'
METADATA_WIKI_FILE = '../../dataset/imdb_crop/wiki_most_famous_actors.pickle'


class Step(Enum):
    RECEIVE_IMAGE = 0
    ONE_FACE_DETECTED = 1
    MULTI_FACE_DETECTED = 2


class TelegramBot:
    #TOKEN = '5085307623:AAEojC_68VWSig4C2Jw5LhC1xuzX76Xtagc'#


    TOKEN = '5029509042:AAE0ji8V8uHWIF_RT5a_qWGqf0qk3uTKrcc'

    def __init__(self):
        # Bot
        self.bot = TelegramBot.init_bot()
        self.step = Step.RECEIVE_IMAGE
        self.faces = []
        # Init Keras sessions
        self.init_keras_session()
        # Init yolo face detector
        #self.init_yolo_face_detector() # <---------- YOLO DETECTOR INIZIALIZZAZIONE
        # Init cascade face detector
        self.init_cascade_face_detector() # <---------- CASCADE DETECTOR INIZIALIZZAZIONE
        # Init VGG model
        self.init_VGG_model()
        # Init similarity
        self.sim = ImageSimilarity(images_path= join('..','..', 'dataset', 'Retrieval', 'images'),
                                   features_path= join('..','..', 'dataset', 'CELEBS', 'features.pickle'),
                                   metadata_path= join('..','..', 'dataset', 'Retrieval', 'wiki_final.pickle')
                                   )
        self.sim.load_features()

        # # TODO: Istanziare due sim invece che uno, uno con features_male.pickle e uno con features_female.pickle
        # self.sim_male = ImageSimilarity(images_path= join('..','..', 'dataset', 'Retrieval', 'images'),
        #                                    features_path= join('..','..', 'dataset', 'CELEBS', 'features_male.pickle'),
        #                                    metadata_path= join('..','..', 'dataset', 'Retrieval', 'wiki_final.pickle')
        #                                    )
        # self.sim_female = ImageSimilarity(images_path= join('..','..', 'dataset', 'Retrieval', 'images'),
        #                                    features_path= join('..','..', 'dataset', 'CELEBS', 'features_female.pickle'),
        #                                    metadata_path= join('..','..', 'dataset', 'Retrieval', 'wiki_final.pickle')
        #


    def init_keras_session(self):
        self.graph = tf.compat.v1.get_default_graph()
        self.session = None

    def init_yolo_face_detector(self,
                                model_path='../../model/yolo_finetuned_best.h5',
                                classes_path='../../libs/yolo3_keras/model_data/fddb_classes.txt',
                                anchors_path='../../libs/yolo3_keras/model_data/yolo_anchors.txt'):
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph)
            with self.session.as_default():
                self.yolo_face_detector = YoloFaceDetector(model_path=model_path,
                                                           classes_path=classes_path,
                                                           anchors_path=anchors_path)

    def init_cascade_face_detector(self,
                                    model_path = '../detection/cascade/model/faceDetector_FDDB_LBP_10_0.01.xml'):
            self.cascade_face_detector = CascadeFaceDetector(model_path = model_path)


    def init_VGG_model(self,
                       vggface_path='../../model/vggface_model_final.h5'):
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph)
            with self.session.as_default():
                model = keras.models.load_model(os.path.abspath(vggface_path))

                self.vggface_extractor = keras.Model(inputs=model.layers[1].input,
                                                outputs=model.layers[1].output)

                self.vggface_classif_gender = keras.Model(inputs=model.layers[2].input,
                                              outputs=model.layers[2].output)
                self.vggface_classif_age = keras.Model(inputs=model.layers[3].input,
                                              outputs=model.layers[3].output)

    def start_main_loop(self):
        # Start mainloop
        print('Listening ...')
        self.bot.message_loop(self.on_chat_message)

    def on_chat_message(self, msg):
        content_type, chat_type, chat_id = telepot.glance(msg)  # get dei parametri della conversazione e del tipo di messaggio

        if self.step == Step.RECEIVE_IMAGE:
            if content_type == 'text':
                name = msg["from"]["first_name"]
                txt = msg['text']

                if txt == '/start':
                    self.bot.sendMessage(chat_id, 'Ciao %s, benvenuto!\nInserisci una foto raffigurante una persona per inziare' % name)

            if content_type == 'photo':
                self.bot.download_file(msg['photo'][-1]['file_id'], 'received_image.png')
                img = cv2.imread('received_image.png')

                # scale_factor = 512/max(img.shape)
                # dim = (round(img.shape[1]*scale_factor), round(img.shape[0]*scale_factor))

                # img_rescaled = cv2.resize(img, dim)
                img_rescaled = img

                # enhancement
                utils = EnhancementUtils()
                img_rescaled = utils.adaptive_gamma(img_rescaled)
                img_rescaled = utils.bilateral_filter(img_rescaled)

                self.bot.sendMessage(chat_id, 'Sto analizzando la foto...')

                # self.faces = self.detect_faces_yolo(img_rescaled)# <---------- YOLO DETECTOR DETECTION
                self.faces = self.cascade_face_detector.detect_image(img_rescaled) # <---------- CASCADE DETECTOR DETECTION
                print(self.faces)

                num_faces_found = len(self.faces)

                print(f'Num faces found', num_faces_found)
                if num_faces_found != 0:
                    for idx, (x_min, y_min, x_max, y_max) in enumerate(self.faces):
                        label = str(idx + 1)
                        cv2.rectangle(img_rescaled, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

                        (w_space, h_space), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(img_rescaled, (x_min, y_min - 20), (x_min + w_space, y_min), (0, 0, 255), -1)
                        cv2.putText(img_rescaled, label, (x_min, y_min - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    detected_imagepath = 'detected_image.png'
                    cv2.imwrite(detected_imagepath, img_rescaled)

                    self.bot.sendPhoto(chat_id, photo=open(detected_imagepath, 'rb'))

                    if num_faces_found == 1:
                        self.bot.sendMessage(chat_id, 'Nella foto è stato individuato un volto! Desideri confermare?\n/Conferma\n/Annulla')
                        self.step = Step.ONE_FACE_DETECTED
                        print(f'Step: {self.step}')
                    elif num_faces_found > 1:
                        self.bot.sendMessage(chat_id, 'Nella foto è stato individuato più di un volto! Quale desideri utilizzare?\n(inserisci il numero del volto scelto)\n/Annulla')
                        self.step = Step.MULTI_FACE_DETECTED
                        print(f'Step: {self.step}')
                else:
                    self.bot.sendMessage(chat_id, 'Nella foto non sono stati rilevati volti, inviare una nuova foto')

        if self.step == Step.ONE_FACE_DETECTED:
            if content_type == 'text':
                name = msg["from"]["first_name"]
                txt = msg['text']

                if txt == '/Conferma':
                    x_min, y_min, x_max, y_max = self.faces[0]

                    print(f'Sel Face: ', self.faces[0])

                    img = cv2.imread('received_image.png')

                    cropped_img = img[y_min:y_max, x_min:x_max]

                    # Predizioni con la rete
                    features, predicted_age_min, predicted_age_max, predicted_gender = self.make_vgg_predictions(cropped_img)

                    print(f'Genere predetto: {predicted_gender}')
                    print(f'Età predetta: {predicted_age_min}-{predicted_age_max}')

                    gender_dict = {0: 'Maschio',
                                   1: 'Femmina'}

                    self.bot.sendMessage(chat_id,
                                         'Genere predetto: ' + gender_dict[
                                             predicted_gender] + '\nEtà predetta: [' + str(
                                             predicted_age_min) + ';' + str(predicted_age_max) + ']')

                    #Perform retrieval of most similar celebrity
                    celeb_name, celeb_image_path, celeb_dist = self.retrieve_similar_celeb(features, predicted_gender,
                                                                               predicted_age_min+5)

                    self.bot.sendPhoto(chat_id, photo=open(celeb_image_path, 'rb'),
                                       caption='Caspita! Assomigli proprio a ' + celeb_name + '\nSomiglianza: ' +
                                               str(round(celeb_dist, 2)))


                    self.step = Step.RECEIVE_IMAGE
                    print(f'Step: {self.step}')

                elif txt == '/Annulla':
                    self.faces = []
                    self.step = Step.RECEIVE_IMAGE
                    print(f'Step: {self.step}')
                else:
                    self.bot.sendMessage(chat_id, 'Input non valido, riprovare.\n/Conferma\n/Annulla')

        if self.step == Step.MULTI_FACE_DETECTED:
            if content_type == 'text':
                name = msg["from"]["first_name"]
                txt = msg['text']

                if txt == '/Annulla':
                    self.faces = []
                    self.step = Step.RECEIVE_IMAGE
                    print(f'Step: {self.step}')
                elif txt.isnumeric():
                    idx = int(txt) - 1
                    if idx <= len(self.faces):
                        x_min, y_min, x_max, y_max = self.faces[idx]
                        print(f'Sel face: ', self.faces[idx])
                        print(x_min,y_min,x_max,y_max)
                        img = cv2.imread('received_image.png')

                        cropped_img = img[y_min:y_max, x_min:x_max]

                        # Predizioni con la rete
                        #predicted_age, predicted_gender = make_vgg_predictions(gender_vgg_model,age_vgg_model,cropped_img)

                        features, predicted_age_min, predicted_age_max, predicted_gender = self.make_vgg_predictions(cropped_img)

                        print(f'Genere predetto: {predicted_gender}')
                        print(f'Età predetta: {predicted_age_min}-{predicted_age_max}')


                        gender_dict = {0: 'Maschio',
                                       1: 'Femmina'}

                        self.bot.sendMessage(chat_id,
                                        'Genere predetto: ' + gender_dict[predicted_gender] + '\nEtà predetta: [' + str(
                                            predicted_age_min) + ';' + str(predicted_age_max) + ']')

                        # Perform retrieval of most similar celebrity
                        celeb_name, celeb_image_path, celeb_dist = self.retrieve_similar_celeb(features, predicted_gender,
                                                                                   predicted_age_min+5)

                        self.bot.sendPhoto(chat_id, photo=open(celeb_image_path, 'rb'),
                                           caption='Caspita! Assomigli proprio a ' + celeb_name + '\nSomiglianza: ' +
                                                   str(round(celeb_dist, 2)))

                        self.step = Step.RECEIVE_IMAGE
                        print(f'Step: {self.step}')

                else:
                    self.bot.sendMessage(chat_id, 'Input non valido, riprovare.\n(inserisci il numero del volto scelto)\n/Annulla')

    def make_vgg_predictions(self, img):
        SCALER = 116

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img /= 255.0
        img = cv2.resize(img, (224, 224))

        cv2.imwrite('cropped_and_preprocessed_image.png', 255*img)


        with self.graph.as_default():
            with self.session.as_default():
                img = np.expand_dims(img, 0)

                # Prediction
                features = self.vggface_extractor.predict(img)
                prediction_gender = self.vggface_classif_gender.predict(features)
                prediction_age = self.vggface_classif_age.predict(features)

                #prediction_gender = pred[0]
                #prediction_age = pred[1]

                prediction_gender = np.argmax(prediction_gender)
                prediction_age = round(float(prediction_age * SCALER))

                # Gender
                #prediction_gender = self.gender_vgg_model.predict(img)
                #prediction_gender = np.argmax(prediction_gender)
                # Age
                #prediction_age = self.age_vgg_model.predict(img)
                #prediction_age = round(float(prediction_age*SCALER))

        return features, max(0, prediction_age-5), prediction_age+5, prediction_gender

    @staticmethod
    def init_bot():
        bot = telepot.Bot(TelegramBot.TOKEN)
        bot.setWebhook()  # unset webhook by supplying no parameter
        return bot

    def detect_faces_yolo(self, img_rescaled):
        with self.graph.as_default():
            with self.session.as_default():
                img = Image.fromarray(img_rescaled)
                return self.yolo_face_detector.detect_image(img, return_confidence=False, th=0.5)

    def retrieve_similar_celeb(self, features, predicted_gender, predicted_age):

        _, most_similar_actor, dist = self.sim.find_most_similar(features)

        # if predicted_gender == 0:
        #     _, most_similar_actor, dist = self.sim_male.find_most_similar(features)
        # else:
        #     _, most_similar_actor, dist = self.sim_female.find_most_similar(features)

        return most_similar_actor.loc['name'], join('../', most_similar_actor.loc['path']), dist

# Loading detector
#cascade_face_detector = CascadeFaceDetector()

'''
gender_vgg_model._make_predict_function()
age_vgg_model._make_predict_function()
'''

bot = TelegramBot()
bot.start_main_loop()

while 1:
    time.sleep(10)









