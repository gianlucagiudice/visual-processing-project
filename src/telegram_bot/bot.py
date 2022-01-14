import os.path

import numpy as np
import telepot
from telepot.loop import MessageLoop
from pprint import pprint
import time
import cv2
from PIL import Image

import keras
import tensorflow as tf


from src.EnhancementUtils import EnhancementUtils
from src.detection.cascade.CascadeFaceDetector import CascadeFaceDetector
from src.detection.yolo.YoloFaceDetector import YoloFaceDetector


TOKEN = '5085307623:AAEojC_68VWSig4C2Jw5LhC1xuzX76Xtagc'

#TOKEN = '5029509042:AAE0ji8V8uHWIF_RT5a_qWGqf0qk3uTKrcc'

FACES = []
step = 0


def on_chat_message(msg):
    global step
    global FACES

    content_type, chat_type, chat_id = telepot.glance(msg)  # get dei parametri della conversazione e del tipo di messaggio

    if step == 0:
        if content_type == 'text':
            name = msg["from"]["first_name"]
            txt = msg['text']

            if txt == '/start':
                bot.sendMessage(chat_id, 'Ciao %s, benvenuto!\nInserisci una foto raffigurante una persona per inziare' % name)

        if content_type == 'photo':
            bot.download_file(msg['photo'][-1]['file_id'], 'received_image.png')
            img = cv2.imread('received_image.png')

            #scale_factor = 512/max(img.shape)
            #dim = (round(img.shape[1]*scale_factor), round(img.shape[0]*scale_factor))

            #img_rescaled = cv2.resize(img, dim)
            img_rescaled = img

            utils = EnhancementUtils()
            if utils.is_image_too_dark(img):
                img_rescaled = utils.automatic_gamma(img_rescaled)

            bot.sendMessage(chat_id, 'Sto analizzando la foto...')

            with graph.as_default():
                with session.as_default():
                    FACES = yolo_face_detector.detect_image(Image.fromarray(img_rescaled))
            print(FACES)

            num_faces_found = len(FACES)

            print(f'Num faces found', num_faces_found)
            if num_faces_found != 0:
                for idx, (x, y, w, h) in enumerate(FACES):
                    label = str(idx + 1)
                    cv2.rectangle(img_rescaled, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    (w_space, h_space), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(img_rescaled, (x, y - 20), (x + w_space, y), (0, 0, 255), -1)
                    cv2.putText(img_rescaled, label, (x, y - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                detected_imagepath = 'detected_image.png'
                cv2.imwrite(detected_imagepath, img_rescaled)

                bot.sendPhoto(chat_id, photo=open(detected_imagepath, 'rb'))

                if num_faces_found == 1:
                    bot.sendMessage(chat_id, 'Nella foto è stato individuato un volto! Desideri confermare?\n/Conferma\n/Annulla')
                    step = 1
                    print(f'Step: {step}')
                elif num_faces_found > 1:
                    bot.sendMessage(chat_id, 'Nella foto è stato individuato più di un volto! Quale desideri utilizzare?\n(inserisci il numero del volto scelto)\n/Annulla')
                    step = 2
                    print(f'Step: {step}')
            else:
                bot.sendMessage(chat_id, 'Nella foto non sono stati rilevati volti, inviare una nuova foto')

    if step == 1:
        if content_type == 'text':
            name = msg["from"]["first_name"]
            txt = msg['text']

            if txt == '/Conferma':
                x, y, w, h = FACES[0]

                print(f'Sel Face: ', FACES[0])

                img = cv2.imread('received_image.png')

                cropped_img = img[y:y + h, x:x + w]

                # Predizioni con la rete
                predicted_age, predicted_gender = make_vgg_predictions(gender_vgg_model, age_vgg_model, cropped_img)

                print(f'Genere predetto: {predicted_gender}')
                print(f'Età esatta predetta: {predicted_age}')

                gender_dict = {0: 'Maschio',
                               1: 'Femmina'}

                bot.sendMessage(chat_id,
                                'Genere predetto: ' + gender_dict[predicted_gender] + '\nEtà predetta: [' + str(
                                    predicted_age - 5) + ';' + str(predicted_age + 5))

                # TODO : perform retrieval of most similar celebrity
                #  celeb_name,celeb_image_path = retrieve_similar_celeb(cropped_img)

                celeb_name = 'Johnny Sins'
                celeb_image_path = 'Johnny Sins.jpg'

                bot.sendPhoto(chat_id, photo=open(celeb_image_path, 'rb'),
                              caption='Caspita! Assomigli proprio a ' + celeb_name)

                step = 0
                print(f'Step: {step}')

            elif txt == '/Annulla':
                FACES = []
                step = 0
                print(f'Step: {step}')
            else:
                bot.sendMessage(chat_id, 'Input non valido, riprovare.\n/Conferma\n/Annulla')

    if step == 2:
        if content_type == 'text':
            name = msg["from"]["first_name"]
            txt = msg['text']

            if txt == '/Annulla':
                FACES = []
                step = 0
                print(f'Step: {step}')
            elif txt.isnumeric():
                idx = int(txt) - 1
                if idx <= len(FACES):
                    x, y, w, h = FACES[idx]
                    print(f'Sel face: ', FACES[idx])
                    print(x,y,w,h)
                    img = cv2.imread('received_image.png')

                    cropped_img = img[y:y+h, x:x+w]

                    # Predizioni con la rete
                    predicted_age, predicted_gender = make_vgg_predictions(gender_vgg_model,age_vgg_model,cropped_img)


                    print(f'Genere predetto: {predicted_gender}')
                    print(f'Età esatta predetta: {predicted_age}')


                    gender_dict = {0: 'Maschio',
                                   1: 'Femmina'}

                    bot.sendMessage(chat_id,
                                    'Genere predetto: ' + gender_dict[predicted_gender] + '\nEtà predetta: [' + str(
                                        predicted_age-5) + ';' + str(predicted_age+5))

                    # TODO : perform retrieval of most similar celebrity
                    #  celeb_name,celeb_image_path = retrieve_similar_celeb(cropped_img)
                    '''
                    celeb_name = 'Johnny Sins'
                    celeb_image_path = 'Johnny Sins.jpg'

                    bot.sendPhoto(chat_id, photo=open(celeb_image_path, 'rb'),
                                  caption='Caspita! Assomigli proprio a ' + celeb_name)
                    '''
                    step = 0
                    print(f'Step: {step}')

            else:
                bot.sendMessage(chat_id, 'Input non valido, riprovare.\n(inserisci il numero del volto scelto)\n/Annulla')

def make_vgg_predictions(model_gender, model_age, img):
    SCALER = 116

    img = cv2.resize(img, (224,224))
    img = img.astype('float32')
    img /= 255.0

    # NON FUNZIONANO:
    with graph.as_default():
        prediction_gender = model_gender.predict(np.expand_dims(img, axis=0))
        prediction_gender = np.argmax(prediction_gender)

    prediction_age = model_age.predict(np.expand_dims(img, axis=0))
    prediction_age = round(float(prediction_age*SCALER))

    return prediction_gender, prediction_age


bot = telepot.Bot(TOKEN)
bot.setWebhook()  # unset webhook by supplying no parameter
bot.message_loop(on_chat_message)
print('Listening ...')

# Loading detector
#cascade_face_detector = CascadeFaceDetector()

global graph
import tensorflow.compat.v1 as tf

graph = tf.compat.v1.get_default_graph()

global session

global yolo_face_detector

with graph.as_default():
    session = tf.Session(graph=graph)
    with session.as_default():

        yolo_face_detector = YoloFaceDetector(model_path='../../model/yolo_finetuned_best.h5',
                                              classes_path='../../libs/yolo3_keras/model_data/fddb_classes.txt',
                                              anchors_path='../../libs/yolo3_keras/model_data/yolo_anchors.txt')

#keras.backend.clear_session()

'''
gender_vgg_model = keras.models.load_model(os.path.abspath('../../model/finetuned_vgg_gender_best.h5'))
age_vgg_model = keras.models.load_model(os.path.abspath('../../model/finetuned_vgg_age_best.h5'))

gender_vgg_model._make_predict_function()
age_vgg_model._make_predict_function()
'''


print(f'Step: {0}')

while 1:
    time.sleep(10)









