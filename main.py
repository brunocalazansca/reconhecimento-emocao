import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

imagem = cv2.imread('Material/Material/testes/eu.jpeg')

cascade_faces = 'Material/Material/haarcascade_frontalface_default.xml'
caminho_modelo = 'Material/Material/modelo_01_expressoes.h5'
face_detection = cv2.CascadeClassifier(cascade_faces)
classificador_emocoes = load_model(caminho_modelo, compile=False)
expressoes = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']

original = imagem.copy()
cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

faces = face_detection.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

probabilidades = np.ones((250, 300, 3), dtype=np.uint8) * 255

if len(faces) > 0:
    (x, y, w, h) = faces[0]
    roi = cinza[y:y + h, x:x + w]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = classificador_emocoes.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = expressoes[preds.argmax()]

    cv2.putText(original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for (i, (emotion, prob)) in enumerate(zip(expressoes, preds)):
        texto = '{}: {:.2f}%'.format(emotion, prob * 100)
        largura_barra = int(prob * 300)
        cv2.rectangle(probabilidades, (7, (i * 35) + 5), (largura_barra, (i * 35) + 35), (200, 250, 20), -1)
        cv2.putText(probabilidades, texto, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

cv2.imshow('Imagem original', original)
cv2.imshow('Probabilidades', probabilidades)
cv2.waitKey(0)
cv2.destroyAllWindows()