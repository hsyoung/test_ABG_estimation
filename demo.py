import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from pathlib import Path
import os
import cv2
import dlib
from mtcnn.mtcnn import MTCNN
import numpy as np
import argparse
from contextlib import contextmanager

from tensorflow.keras.preprocessing import image

from src.abg_model import get_model


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_func", type=str, default=None,
                        help="argument indicating the model-function to demo !\n"
                             "(bmi-face, age-face, gender-face, abg-face), exclusively ")
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.***.hdf5)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()

def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_CUBIC)


def main():
    args = get_args()
    model_func = args.model_func
    weight_file = args.weight_file
    image_dir = args.image_dir
    margin = 0.4

    # DEBUG ...
    print("hello!\n"
          "-------\n")

    if not weight_file:
        print("missing of path to pretrained-model-file! \n"
              "in the future, the pretrained-model-file can be downloaded from URL if not locally being\n"
              "checked ready !\n ")
        return
    # -------------------
    # for face detection
    # -------------------
    # detector_dlib = dlib.get_frontal_face_detector()
    detector_mtcnn = MTCNN()

    # ----------------------------------
    # Choose "model-function" to DEMO
    # ---, load model and weights
    if model_func == "bmi-face":
        model = get_model("bmi-face")
        model.load_weights(os.path.join(os.getcwd() + "/" + weight_file + "/" + "bmi_model_weights.h5"))
    elif model_func == "age-face":
        model = get_model("age-face")
        model.load_weights(os.path.join(os.getcwd() + "/" + weight_file + "/" + "age_model_weights.h5"))
        ages_indexes = np.array([i for i in range(0, 101)])
    elif model_func == "gender-face":
        model = get_model("gender-face")
        model.load_weights(os.path.join(os.getcwd() + "/" + weight_file + "/" + "gender_model_weights.h5"))
    elif model_func == "age-bmi-gender-face":
        model = get_model("abg-face")
        model.load_weights(os.path.join(os.getcwd() + "/" + weight_file + "/" + "abg_model_weights.h5"))
        ages_indexes = np.array([i for i in range(0, 101)])

    else:
        print("Please choose within (bmi-face, age-face, gender-face, abg-face), exclusively!\n")
        return

    # ----------------------------------------------------
    # process the faces in images or video from webcam
    # [Attention]:
    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()

    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # ------------------------------------
        # detect faces
        # ------------------------------------
        # using dlib detector
        # detected_dlib = detector_dlib(input_img, 1)
        # faces = np.empty((len(detected_dlib), 224, 224, 3))

        # using mtcnn detector
        detected = detector_mtcnn.detect_faces(input_img)
        faces = np.empty((len(detected), 224, 224, 3))

        if len(detected) > 0:
            """"
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)

                detected_face = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1], (224, 224),
                                            interpolation=cv2.INTER_CUBIC)
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
                faces[i] = img_pixels
            """
            for i, d in enumerate(detected):
                bbox = d['box']
                # x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                x1, y1, x2, y2, w, h = bbox[0], bbox[1], bbox[0]+bbox[2]+1, bbox[1]+bbox[3]+1, bbox[2], bbox[3]

                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)

                detected_face = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1], (224, 224),
                                           interpolation=cv2.INTER_CUBIC)
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
                faces[i] = img_pixels
            # --------------------------------------------------
            # predict ages/genders/bmi of the detected faces
            if model_func == "age-face":
                age_distributions = model.predict(faces)
                predicted_ages = age_distributions.dot(ages_indexes).flatten()
                # draw results
                for i, d in enumerate(detected):
                    bbox = d['box']
                    label: str = "Age: {}".format(int(predicted_ages[i]))
                    # draw_label(img, (d.left(), d.top()), label)
                    draw_label(img, (bbox[0], bbox[1]), label)
            elif model_func == "gender-face":
                predicted_genders = model.predict(faces)
                # draw results
                for i, d in enumerate(detected):
                    bbox = d['box']
                    label = "{}".format("M" if predicted_genders[i][0] < 0.5 else "F")
                    # draw_label(img, (d.left(), d.top()), label)
                    draw_label(img, (bbox[0], bbox[1]), label)
            elif model_func == "bmi-face":
                results = model.predict(faces)
                predicted_bmi = results
                # draw results
                for i, d in enumerate(detected):
                    bbox = d['box']
                    label = "BMI:" + "{}".format(int(predicted_bmi[i]))
                    # draw_label(img, (d.left(), d.top()), label)
                    draw_label(img, (bbox[0], bbox[1]+bbox[3]), label)
            elif model_func == "abg-face":
                    results = model.predict(faces)
                    predicted_genders = results[0]
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_ages = results[1].dot(ages).flatten()
                    predicted_bmi = results[2]
                    # draw results
                    for i, d in enumerate(detected):
                        bbox = d['box']
                        label = "{}, {}, {}".format(int(predicted_ages[i]),
                                                    "M" if predicted_genders[i][0] < 0.5 else "F",
                                                    float(predicted_bmi[i]))
                        # draw_label(img, (d.left(), d.top()), label)
                    draw_label(img, (bbox[0], bbox[1]), label)

        cv2.imshow("result", img)
        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        if key == 27:  # ESC
            break


if __name__ == '__main__':
    main()
