import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from tensorflow.keras.models  import model_from_json
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import glob
import easyocr
import pytesseract
from PIL import Image
import imutils


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Model Loaded successfully")
        return model
    except Exception as e:
        print(e)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_path, wpod_net, Dmax=608, Dmin = 608):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def plate_recognitionCNN(file):
    wpod_net_path = "./models/wpod-net.json"
    wpod_net = load_model(wpod_net_path)

    test_image_path = file
    vehicle, LpImg, cor = get_plate(test_image_path, wpod_net)

    # fig = plt.figure(figsize=(12,6))
    # grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    # fig.add_subplot(grid[0])
    # plt.axis(False)
    # plt.imshow(vehicle)
    # grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    # fig.add_subplot(grid[1])
    # plt.axis(False)
    # plt.imshow(LpImg[0])

    img = interval_mapping(LpImg[0], 0.0, 1.0, 0, 255)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(np.uint8(img))

    finalResult = ""

    for i in range (len(result)):
        finalResult += result[i][1]
        if i == 2:
            break
    
    img = imutils.resize(np.uint8(img), width=250)
    return img, finalResult
    