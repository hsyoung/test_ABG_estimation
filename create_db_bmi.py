import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import scipy.io
from tqdm import tqdm

from collections import  OrderedDict

__all__ = [cv2]


def get_args():
    parser = argparse.ArgumentParser(description="This script creates database for training including \
                                        labels of samples from the folder bmi_face_data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to the directory of BMI image of cropped face ")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="absolute path of label file, in matlab format, into folder bmi_face_data")
    parser.add_argument("--img_size", type=int, default=64,
                        help="image size for training samples")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    image_dir = Path(args.input)
    output_path = os.path.join(os.getcwd() + "/" + args.output)
    img_size = args.img_size

    out_genders = []
    out_ages = []
    out_height = []
    out_weight = []
    out_bmi = []
    out_db = []
    out_img_size = []
    out_img_path = []

    for i, image_path in enumerate(tqdm(image_dir.glob("**/*.jpg"), disable=None)):
        image_name = image_path.name  # [Gender]-[Age]-[Height]-[Weight]-[Random UUID].jpg
        gender, age, height, weight = image_name.split("-")[:4]
        if gender == 'F':
            out_genders.append(1)
        elif gender == 'M':
            out_genders.append(0)
        else:
            continue
        out_ages.append(min(int(age), 100))
        out_height.append(float(height))
        out_weight.append(float(weight))

        bmi = float(weight)/(float(height)**2)
        out_bmi.append(bmi)

        out_img_size.append(img_size)

        out_img_path.append(str(image_path))
        #img = cv2.imread(str(image_path))
        #out_imgs.append(cv2.resize(img, (img_size, img_size)))

    output = dict(OrderedDict([("gender", np.array(out_genders)),
                        ("age", np.array(out_ages)), ("height", np.array(out_height)),
                        ("weight", np.array(out_weight)), ("bmi", np.array(out_bmi)),
                        ("img_size", np.array(out_img_size)),
                        ("img_path", np.array(out_img_path, dtype=object))]))
    # ("img_path", np.array(out_img_path))
    scipy.io.savemat(os.path.join(output_path + "/" + "bmi_train_data.mat"), output)


if __name__ == '__main__':
    main()
