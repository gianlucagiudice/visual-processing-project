import os.path
from enum import Enum

import numpy as np
import telepot
import time
import cv2
from PIL import Image

import keras
import tensorflow.compat.v1 as tf

from src.EnhancementUtils import EnhancementUtils
from src.detection.yolo.YoloFaceDetector import YoloFaceDetector


class Step(Enum):
    RECEIVE_IMAGE = 0
    ONE_FACE_DETECTED = 1
    MULTI_FACE_DETECTED = 2


class TelegramBot:
    TOKEN = '5085307623:AAEojC_68VWSig4C2Jw5LhC1xuzX76Xtagc'

    def __init__(self):
        # Bot
        self.bot = TelegramBot.init_bot()
        self.step = Step.RECEIVE_IMAGE
        self.faces = []
        # Init Keras sessions
        self.init_keras_session()
        # Init yolo face detector
        self.init_yolo_face_detector()
        # Init VGG model
        self.init_VGG_model()

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

    def init_VGG_model(self,
                       vgg_gender_path='../../model/finetuned_vgg_gender_best.h5',
                       vgg_age_path='../../model/finetuned_vgg_age_best.h5'):
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph)
            with self.session.as_default():
                self.gender_vgg_model = keras.models.load_model(os.path.abspath(vgg_gender_path))
                self.age_vgg_model = keras.models.load_model(os.path.abspath(vgg_age_path))

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

                #scale_factor = 512/max(img.shape)
                #dim = (round(img.shape[1]*scale_factor), round(img.shape[0]*scale_factor))

                #img_rescaled = cv2.resize(img, dim)
                img_rescaled = img

                utils = EnhancementUtils()
                if utils.is_image_too_dark(img):
                    img_rescaled = utils.equalize_histogram(np.uint8(img_rescaled * 255))
                    img_rescaled = utils.automatic_gamma(img_rescaled)
                    img_rescaled = utils.adaptive_gamma(img_rescaled)

                self.bot.sendMessage(chat_id, 'Sto analizzando la foto...')

                with self.graph.as_default():
                    with self.session.as_default():
                        self.faces = self.yolo_face_detector.detect_image(Image.fromarray(img_rescaled))
                print(self.faces)

                num_faces_found = len(self.faces)

                print(f'Num faces found', num_faces_found)
                if num_faces_found != 0:
                    for idx, (x, y, w, h) in enumerate(self.faces):
                        label = str(idx + 1)
                        cv2.rectangle(img_rescaled, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        (w_space, h_space), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(img_rescaled, (x, y - 20), (x + w_space, y), (0, 0, 255), -1)
                        cv2.putText(img_rescaled, label, (x, y - 5),
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
                    x, y, w, h = self.faces[0]

                    print(f'Sel Face: ', self.faces[0])

                    img = cv2.imread('received_image.png')

                    cropped_img = img[y:y + h, x:x + w]

                    # Predizioni con la rete
                    predicted_age, predicted_gender = make_vgg_predictions(gender_vgg_model, age_vgg_model, cropped_img)

                    print(f'Genere predetto: {predicted_gender}')
                    print(f'Età esatta predetta: {predicted_age}')

                    gender_dict = {0: 'Maschio',
                                   1: 'Femmina'}

                    self.bot.sendMessage(chat_id,
                                    'Genere predetto: ' + gender_dict[predicted_gender] + '\nEtà predetta: [' + str(
                                        predicted_age - 5) + ';' + str(predicted_age + 5))

                    # TODO : perform retrieval of most similar celebrity
                    #  celeb_name,celeb_image_path = retrieve_similar_celeb(cropped_img)

                    celeb_name = 'Johnny Sins'
                    celeb_image_path = 'Johnny Sins.jpg'

                    self.bot.sendPhoto(chat_id, photo=open(celeb_image_path, 'rb'),
                                  caption='Caspita! Assomigli proprio a ' + celeb_name)

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
                        x, y, w, h = self.faces[idx]
                        print(f'Sel face: ', self.faces[idx])
                        print(x,y,w,h)
                        img = cv2.imread('received_image.png')

                        cropped_img = img[y:y+h, x:x+w]

                        # Predizioni con la rete
                        #predicted_age, predicted_gender = make_vgg_predictions(gender_vgg_model,age_vgg_model,cropped_img)
                        # TODO: CAMBIARE QUI!!!!!
                        predicted_age, predicted_gender = 50, 0


                        print(f'Genere predetto: {predicted_gender}')
                        print(f'Età esatta predetta: {predicted_age}')


                        gender_dict = {0: 'Maschio',
                                       1: 'Femmina'}

                        self.bot.sendMessage(chat_id,
                                        'Genere predetto: ' + gender_dict[predicted_gender] + '\nEtà predetta: [' + str(
                                            predicted_age-5) + ';' + str(predicted_age+5))

                        # TODO : perform retrieval of most similar celebrity
                        #  celeb_name,celeb_image_path = retrieve_similar_celeb(cropped_img)
                        '''
                        celeb_name = 'Johnny Sins'
                        celeb_image_path = 'Johnny Sins.jpg'
    
                        self.bot.sendPhoto(chat_id, photo=open(celeb_image_path, 'rb'),
                                      caption='Caspita! Assomigli proprio a ' + celeb_name)
                        '''
                        self.step = Step.RECEIVE_IMAGE
                        print(f'Step: {self.step}')

                else:
                    self.bot.sendMessage(chat_id, 'Input non valido, riprovare.\n(inserisci il numero del volto scelto)\n/Annulla')

    def make_vgg_predictions(self, model_gender, model_age, img):
        SCALER = 116

        img = cv2.resize(img, (224,224))
        img = img.astype('float32')
        img /= 255.0

        # NON FUNZIONANO:
        with self.graph.as_default():
            prediction_gender = model_gender.predict(np.expand_dims(img, axis=0))
            prediction_gender = np.argmax(prediction_gender)

        prediction_age = model_age.predict(np.expand_dims(img, axis=0))
        prediction_age = round(float(prediction_age*SCALER))

        return prediction_gender, prediction_age

    @staticmethod
    def init_bot():
        bot = telepot.Bot(TelegramBot.TOKEN)
        bot.setWebhook()  # unset webhook by supplying no parameter
        return bot


# Loading detector
#cascade_face_detector = CascadeFaceDetector()

'''
gender_vgg_model = keras.models.load_model(os.path.abspath('../../model/finetuned_vgg_gender_best.h5'))
age_vgg_model = keras.models.load_model(os.path.abspath('../../model/finetuned_vgg_age_best.h5'))

gender_vgg_model._make_predict_function()
age_vgg_model._make_predict_function()
'''

bot = TelegramBot()
bot.start_main_loop()

while 1:
    time.sleep(10)









