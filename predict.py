import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from termcolor import colored

model = tf.keras.models.load_model('trained_models/resnet.keras')

def img_pred(image_path):
    try:
        img = Image.open(image_path)
        opencvimage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(opencvimage, (150, 150))
        img = img.reshape(1, 150, 150, 3)
        p = model.predict(img)
        p = np.argmax(p, axis=1)[0]

        if p == 0:
            print(colored('Prediction: Glioma Tumour', 'green', attrs=['bold']))
        elif p == 1:
            print(colored('Prediction: No tumour', 'green', attrs=['bold']))
        elif p == 2:
            print(colored('Prediction: Meningioma Tumour', 'green', attrs=['bold']))
        else:
            print(colored('Prediction: Pituitary Tumour', 'green', attrs=['bold']))
    except Exception as e:
        print(colored(e, 'red', attrs=['bold']))

# Example usage:
img_path = 'test_images/glioma_tumour/1.jpg'

img_pred(img_path)
