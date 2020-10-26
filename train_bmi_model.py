import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from pathlib import Path
import scipy.io
import argparse
import os

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import layers
from tensorflow.python.keras import models

from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

from src.abg_model import get_model


def get_args():
    parser = argparse.ArgumentParser(description="This script trains ABG model based on trained\
                                        model of 'age-gender-estimator' ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bmi_dataset", type=str, default=None,
                        help="path to training dataset ready for direct training process")

    args = parser.parse_args()
    return args

def getImagePixels(image_path):
    target_size = (224, 224)
    img = image.load_img(os.path.join(os.getcwd() + "/" + image_path[0]),
                         grayscale=False, target_size=target_size)
    x = image.img_to_array(img).reshape(1, -1)[0]
    # x = preprocess_input(x)
    return x

def main():
    args = get_args()
    bmi_face_data_path = args.bmi_dataset

    training_path = os.path.join(os.getcwd() + "/" + bmi_face_data_path)

    # -----------------------------
    #   DataSource preparation
    #   --- BMI dataset
    # -----------------------------
    # Converting training data into pandas DataFrame which is used as DataSource
    data_mat = scipy.io.loadmat(os.path.join(training_path + "/" + "bmi_train_data.mat"))
    instances = data_mat["age"].shape[1]
    columns = ["gender", "age", "height", "weight", "bmi", "img_size", "img_path"]
    df = pd.DataFrame(index=range(0, instances), columns=columns)

    j = 0
    for i in data_mat:
        if j >= 3:
            current_array = data_mat[i]
            df[columns[j-3]] = pd.DataFrame(current_array[0])
        j = j + 1
    df = df.drop(columns=['height', 'weight', 'img_size'])

    # Add a "ImgPix" column in the pandas DataFrame which stores the data of image data
    # Add image data into the panda DataFrame table
    df['ImgPix'] = df['img_path'].apply(getImagePixels)

    # for DEBUG.......
    # print(df[['ImgPix']].head(5))
    # print(df.head(5))
    # histogram_age = df['age'].hist(bins=df['age'].nunique())
    # histogram_gender = df['gender'].hist(bins=df['gender'].nunique())

    # ----------------------------------------------------------------------
    #   BMI prediction modeling
    # comment:
    #          1. BMI is one of the three parts in ABG_estimation function
    #             --BMI, as regression problem
    #             --Gender, as binary classification problem
    #             --Age, as "classification+regression" problem
    #          2. various "Backbone CNN model" can serve as backbone network
    #             e.g. VGG, ResNet, EfficientNetB3 etc. for Face Analysis,
    #             including visual BMI-Face, Gender, Age
    #          3. BMI prediction modeling can be got by transfer learning
    #             from ready-for-finetune Face Recognition model
    # ----------------------------------------------------------------------

    # splitting dataset for "Training" and "validation"
    #----------------------------------
    data_face = []
    data_bmi = []
    for i in range(0, df.shape[0]):
        data_face.append(df['ImgPix'].values[i])
        if df['bmi'].values[i] == np.inf:
            print("Error!\n"
                  "The value of -bmi- is np.inf...")
            raise AssertionError
        data_bmi.append(df['bmi'].values[i])

    data_face = np.array(data_face)
    data_bmi = np.array(data_bmi)

    data_face = data_face.reshape(data_face.shape[0], 224, 224, 3)
    # face data normalization
    data_face /= 255
    data_bmi = data_bmi.reshape(data_bmi.shape[0], 1)

    train_x, test_x, train_y, test_y = train_test_split(data_face, data_bmi, test_size=0.30)

    # -------------------------------------------------------------
    # BMI model --- Build from pre-trained BACKBONE VGG-face model
    # -------------------------------------------------------------
    # VGG-Face model
    model = get_model("VGG-face")
    # for DEBUG.......
    # print("the structure of VGG-Face model:\n"
    #      "--------------------------------\n")
    # model.summary()
    # raise AssertionError
    # print("successfully established BACKBONE VGG-faces model ! \n")

    # Load the pre-trained weights of BACKBONE vgg-face model.
    model.load_weights(os.path.join(os.getcwd() + "/bmi_face_model" + "/" + "vgg_face_weights.h5"))

    for layer_in_net in model.layers[:-7]:
        layer_in_net.trainable = False

    # Scheme 1 of bmi_model

    base_bmi_model = models.Sequential()
    base_bmi_model = layers.Convolution2D(160, (1, 1))(model.layers[-4].output)
    base_bmi_model = layers.advanced_activations.PReLU()(base_bmi_model)
    base_bmi_model = layers.Dropout(0.4)(base_bmi_model)
    base_bmi_model = layers.Convolution2D(32, (1, 1))(base_bmi_model)
    base_bmi_model = layers.advanced_activations.PReLU()(base_bmi_model)
    base_bmi_model = layers.Dropout(0.3)(base_bmi_model)
    base_bmi_model = layers.Convolution2D(1, (1, 1), name='predictions_bmi')(base_bmi_model)
    base_bmi_model = layers.Dropout(0.2)(base_bmi_model)
    base_bmi_model = layers.Flatten()(base_bmi_model)

    # Scheme 2 of bmi_model
    """
    base_bmi_model = models.Sequential()
    base_bmi_model = layers.Convolution2D(101, (1, 1), name='ref_agemodel')(model.layers[-4].output)
    base_bmi_model = layers.advanced_activations.ReLU()(base_bmi_model)
    # base_bmi_model = layers.Convolution2D(1, (1, 1), name='predictions_bmi')(base_bmi_model)
    base_bmi_model = layers.Flatten()(base_bmi_model)
    base_bmi_model = layers.Dense(1, kernel_initializer='normal')(base_bmi_model)
    """
    # Scheme 3 of bmi_model
    """
    base_bmi_model = models.Sequential()
    base_bmi_model = layers.Convolution2D(101, (1, 1), name='ref_agemodel')(model.layers[-4].output)
    base_bmi_model = layers.advanced_activations.ReLU()(base_bmi_model)
    base_bmi_model = layers.Convolution2D(1, (1, 1), name='predictions_bmi')(base_bmi_model)
    base_bmi_model = layers.Flatten()(base_bmi_model)
    """
    bmi_model = Model(inputs=model.input, outputs=base_bmi_model)

    # BMI model training......
    # sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam()
    bmi_model.compile(loss='mae', optimizer=adam, metrics=['mse', 'mae'])

    model_checkpoint_callback = ModelCheckpoint(filepath='bmi_model_weights_full.hdf5',
                                   monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    scores = []
    epochs = 250
    batch_size = 128

    for i in range(epochs):
        print("epoch ", i)
        ix_train = np.random.choice(train_x.shape[0], size=batch_size)
        score = bmi_model.fit(train_x[ix_train], train_y[ix_train],
                          epochs=1, validation_data=(test_x, test_y),
                          callbacks=[model_checkpoint_callback])
        scores.append(score)

    bmi_model = load_model("bmi_model_weights_full.hdf5")
    bmi_model.save_weights('bmi_model_weights.h5')

if __name__ == '__main__':
    main()
