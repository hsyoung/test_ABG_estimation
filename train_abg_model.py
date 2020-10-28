import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from pathlib import Path
import argparse
import os
from datetime import datetime, timedelta

import scipy.io
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import tensorflow as tf

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
    parser.add_argument("--wiki_dataset", type=str, default=None,
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

def getImagePixels_wiki(image_path):
    target_size = (224, 224)
    img = image.load_img(os.path.join(os.getcwd() + "/wiki_face_data" + "/wiki_crop/" + image_path[0]),
                         grayscale=False, target_size=target_size)
    x = image.img_to_array(img).reshape(1, -1)[0]
    #x = preprocess_input(x)
    return x

def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    exact_date = datetime.fromordinal(int(datenum)) \
                 + timedelta(days=int(days)) \
                 + timedelta(hours=int(hours)) \
                 + timedelta(minutes=int(minutes)) \
                 + timedelta(seconds=round(seconds)) \
                 - timedelta(days=366)

    return exact_date.year

def main():
    args = get_args()
    bmi_face_data_path = args.bmi_dataset
    wiki_face_data_path = args.wiki_dataset


    training_bmi_path = os.path.join(os.getcwd() + "/" + bmi_face_data_path)

    training_age_gender_data_path = os.path.join(os.getcwd() + "/" + wiki_face_data_path)

    # -----------------------------
    #   DataSource preparation
    #   --- BMI dataset
    # -----------------------------
    # Converting training data into pandas DataFrame which is used as DataSource
    data_mat = scipy.io.loadmat(os.path.join(training_bmi_path + "/" + "bmi_train_data.mat"))
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

    # ----------------------------------------------
    #   DataSource preparation
    #   --- Wikipedia dataset(wiki_crop version)
    #       for Age-Gender prediction
    # ----------------------------------------------
    data_wiki_mat = scipy.io.loadmat(os.path.join(training_age_gender_data_path + "/" + "wiki.mat"))
    instances_wiki = data_wiki_mat['wiki'][0][0][0].shape[1]
    columns_wiki = ["dob", "photo_taken", "full_path", "gender", "name", "face_location",
                  "face_score", "second_face_score"]
    df_wiki = pd.DataFrame(index=range(0, instances_wiki), columns=columns_wiki)

    for i in data_wiki_mat:
        if i == "wiki":
            current_array = data_wiki_mat[i][0][0]
            for j in range(len(current_array)):
                df_wiki[columns_wiki[j]] = pd.DataFrame(current_array[j][0])

    df_wiki['date_of_birth'] = df_wiki['dob'].apply(datenum_to_datetime)
    df_wiki['age'] = df_wiki['photo_taken'] - df_wiki['date_of_birth']

    # data cleaning of original wiki_crop dataset
    df_wiki = df_wiki[df_wiki['face_score'] != -np.inf]
    df_wiki = df_wiki[df_wiki['second_face_score'].isna()]
    df_wiki = df_wiki[df_wiki['face_score'] >= 3]
    df_wiki = df_wiki.drop(columns=['name', 'face_score', 'second_face_score', 'date_of_birth', 'face_location'])
    df_wiki = df_wiki[df_wiki['age'] <= 100]
    df_wiki = df_wiki[df_wiki['age'] > 0]

    # Add a "ImgPix" column in the pandas DataFrame which stores the data of image data
    # Add image data into the panda DataFrame table
    df_wiki['ImgPix'] = df_wiki['full_path'].apply(getImagePixels_wiki)

    # for DEBUG.......
    # print(df_wiki[['ImgPix']].head(5))
    # print(df_wiki.head(5))

    # -----------------------------------------------------
    # splitting dataset for "Training" and "validation"
    # -----------------------------------------------------
    # BMI branch
    #--------------
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

    face_bmi_train_x, face_bmi_test_x, face_bmi_train_y, face_bmi_test_y = train_test_split(
        data_face, data_bmi, test_size=0.30)

    # --------------------------------
    # Age branch
    # -------------
    target = df_wiki['age'].values
    classes = 101  # (0, 100])
    age_classes = tf.keras.utils.to_categorical(target, classes)

    data_wiki_face = []

    for i in range(0, df_wiki.shape[0]):
        data_wiki_face.append(df_wiki['ImgPix'].values[i])

    data_wiki_face = np.array(data_wiki_face)
    data_wiki_face = data_wiki_face.reshape(data_wiki_face.shape[0], 224, 224, 3)
    data_wiki_face /= 255
    # --------------------------------
    # Gender branch
    # ---------------
    target = df_wiki['gender'].values
    gender_classes = tf.keras.utils.to_categorical(target, 2)

    split = train_test_split(data_wiki_face, gender_classes, age_classes,
                             test_size=0.3)
    (face_wiki_train_x, face_wiki_test_x, face_wiki_train_gender_y, face_wiki_test_gender_y,
     face_wiki_train_age_y, face_wiki_test_age_y) = split
    # ----------------------------------------------------------------------
    #   Age-BMI-Gender(ABG) prediction modeling
    # comment:
    #          1. BMI is one of the three parts in ABG_estimation function
    #             --BMI, as regression problem
    #             --Gender, as binary classification problem
    #             --Age, as "classification+regression" problem
    #          2. various "Backbone CNN model" can serve as backbone network
    #             e.g. VGG, ResNet, EfficientNetB3 etc. for Face Analysis,
    #             including visual BMI-Face, Gender, Age
    # ----------------------------------------------------------------------
    # -----------------------------------------------
    # Build from pre-trained BACKBONE VGG-face model
    # -------------------
    # VGG-Face model
    backbone_model = get_model("VGG-face")
    # for DEBUG.......
    # print("the structure of VGG-Face model:\n"
    #      "--------------------------------\n")
    # backbone_model.summary()
    # print("successfully established BACKBONE VGG-faces model ! \n")

    # Load the pre-trained weights of BACKBONE vgg-face model.
    backbone_model.load_weights(os.path.join(os.getcwd() + "/bmi_face_model" + "/" + "vgg_face_weights.h5"))

    for layer_in_net in backbone_model.layers[:-7]:
        layer_in_net.trainable = False

    # --------------------------------
    # BMI branch
    # ---------------
    # Scheme 1 of bmi_model
    base_bmi_model = models.Sequential()
    base_bmi_model = layers.Convolution2D(160, (1, 1))(backbone_model.layers[-4].output)
    base_bmi_model = layers.advanced_activations.PReLU()(base_bmi_model)
    base_bmi_model = layers.Dropout(0.4)(base_bmi_model)
    base_bmi_model = layers.Convolution2D(32, (1, 1))(base_bmi_model)
    base_bmi_model = layers.advanced_activations.PReLU()(base_bmi_model)
    base_bmi_model = layers.Dropout(0.3)(base_bmi_model)
    base_bmi_model = layers.Convolution2D(1, (1, 1), name='predictions_bmi')(base_bmi_model)
    base_bmi_model = layers.Dropout(0.2)(base_bmi_model)
    base_bmi_model = layers.Flatten(name='bmi_output')(base_bmi_model)
    # --------------------------------
    # Age branch
    # ---------------
    base_age_model = models.Sequential()
    base_age_model = layers.Convolution2D(101, (1, 1), name='predictions_age')(backbone_model.layers[-4].output)
    base_age_model = layers.Flatten()(base_age_model)
    base_age_model = layers.Activation('softmax', name='age_output')(base_age_model)
    # --------------------------------
    # Gender branch
    # ---------------
    base_gender_model = models.Sequential()
    base_gender_model = layers.Convolution2D(2, (1, 1), name='predictions_gender')(backbone_model.layers[-4].output)
    base_gender_model = layers.Flatten()(base_gender_model)
    base_gender_model = layers.Activation('softmax', name='gender_output')(base_gender_model)
    # --------------------------------
    # ABG model assembling
    # ----------------------
    abg_model = Model(inputs=backbone_model.input, outputs=[base_gender_model,
                                                   base_age_model,
                                                   base_bmi_model], name='ABG_model')
    # for DEBUG....
    abg_model.summary()
    raise AssertionError
    # ---------------------------
    # ABG model training......
    # ---------------------------
    # sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam()
    abg_model.compile(optimizer=adam,
                      loss={'gender_output': 'categorical_crossentropy',
                            'age_output': 'categorical_crossentropy',
                            'bmi_output': 'mae'},
                      loss_weights={'gender_output': 1,
                                    'age_output': 5,
                                    'bmi_output': 2},
                      metrics={'gender_output': 'accuracy',
                               'age_output': 'accuracy',
                               'bmi_output': 'mse'}
                      )

    model_checkpoint_callback = ModelCheckpoint(filepath='abg_model_weights.hdf5',
                                   monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    scores_age_gender = []
    epochs_wiki = 90
    batch_size_wiki = 64
    scores_bmi = []
    epochs_bmi = 250
    batch_size_bmi = 128

    for i in range(epochs_wiki):
        print("Epoch of training of wiki_dataset... ", i)
        ix_train = np.random.choice(face_bmi_train_x.shape[0], size=batch_size_wiki)
        score_age_gender = abg_model.fit(
                          x=face_wiki_train_x[ix_train],
                          y={"age_output": face_wiki_train_age_y[ix_train],
                             "gender_output": face_wiki_train_gender_y[ix_train]},
                          validation_data=(face_wiki_test_x,
                                       {"age_output": face_wiki_test_age_y,
                                        "gender_output": face_wiki_test_gender_y}),
                          epochs=1,
                          verbose=1,
                          callbacks=[model_checkpoint_callback])

        scores_age_gender.append(score_age_gender)

    for i in range(epochs_bmi):
        print("Epoch of training on bmi_dataset... ", i)
        ix_train = np.random.choice(face_bmi_train_x.shape[0], size=batch_size_bmi)
        score_bmi = abg_model.fit(
                          x=face_bmi_train_x[ix_train],
                          y={"bmi_output": face_bmi_train_y[ix_train]},
                          validation_data=(face_bmi_test_x, face_bmi_test_y),
                          epochs=1,
                          callbacks=[model_checkpoint_callback])
        scores_bmi.append(score_bmi)

    abg_model = load_model("abg_model_weights_full.hdf5")
    abg_model.save_weights('abg_model_weights.h5')


if __name__ == '__main__':
    main()
