
import telepot
from telepot.loop import MessageLoop
from pprint import pprint
import time
import cv2

from src.detection.cascade.CascadeFaceDetector import CascadeFaceDetector

TOKEN = '5085307623:AAEojC_68VWSig4C2Jw5LhC1xuzX76Xtagc'

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

            bot.sendMessage(chat_id, 'Sto analizzando la foto...')

            FACES = cascade_face_detector.detect_image(img_rescaled)
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
                elif num_faces_found > 1:
                    bot.sendMessage(chat_id, 'Nella foto è stato individuato più di un volto! Quale desideri utilizzare?\n(inserisci il numero del volto scelto)\n/Annulla')
                    step = 2
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

                # TODO : perform prediction of age and gender from cropped_img
                #  predicted_age = predict_age(cropped_img)
                #  predicted_gender = predict_gender(cropped_img)

                predicted_age = 60
                predicted_gender = 1

                gender_dict = {1: 'Maschio',
                               0: 'Femmina'}

                bot.sendMessage(chat_id, 'Genere predetto: ' + gender_dict[predicted_gender] + '\nEtà predetta: ' + str(
                    predicted_age))

                # TODO : perform retrieval of most similar celebrity
                #  celeb_name,celeb_image_path = retrieve_similar_celeb(cropped_img)

                celeb_name = 'Johnny Sins'
                celeb_image_path = 'Johnny Sins.jpg'

                bot.sendPhoto(chat_id, photo=open(celeb_image_path, 'rb'),
                              caption='Caspita! Assomigli proprio a ' + celeb_name)

                step = 0

            elif txt == '/Annulla':
                FACES = []
                step = 0
            else:
                bot.sendMessage(chat_id, 'Input non valido, riprovare.\n/Conferma\n/Annulla')

    if step == 2:
        if content_type == 'text':
            name = msg["from"]["first_name"]
            txt = msg['text']

            if txt == '/Annulla':
                FACES = []
                step = 0
            elif txt.isnumeric():
                idx = int(txt) - 1
                if idx <= len(FACES):
                    x, y, w, h = FACES[idx]
                    print(f'Sel face: ', FACES[idx])
                    print(x,y,w,h)
                    img = cv2.imread('received_image.png')

                    cropped_img = img[y:y+h, x:x+w]

                    # TODO : perform prediction of age and gender from cropped_img
                    #  predicted_age = predict_age(cropped_img)
                    #  predicted_gender = predict_gender(cropped_img)

                    predicted_age = 60
                    predicted_gender = 1

                    gender_dict = {1: 'Maschio',
                                   0: 'Femmina'}

                    bot.sendMessage(chat_id,
                                    'Genere predetto: ' + gender_dict[predicted_gender] + '\nEtà predetta: ' + str(
                                        predicted_age))

                    # TODO : perform retrieval of most similar celebrity
                    #  celeb_name,celeb_image_path = retrieve_similar_celeb(cropped_img)

                    celeb_name = 'Johnny Sins'
                    celeb_image_path = 'Johnny Sins.jpg'

                    bot.sendPhoto(chat_id, photo=open(celeb_image_path, 'rb'),
                                  caption='Caspita! Assomigli proprio a ' + celeb_name)

                    step = 0

            else:
                bot.sendMessage(chat_id, 'Input non valido, riprovare.\n(inserisci il numero del volto scelto)\n/Annulla')


bot = telepot.Bot(TOKEN)
bot.message_loop(on_chat_message)
print('Listening ...')
cascade_face_detector = CascadeFaceDetector()


while 1:
    time.sleep(10)









