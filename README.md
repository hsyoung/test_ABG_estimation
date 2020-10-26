# Age-BMI-Gender Estimation
This is a Tensorflow-Keras implementation of a CNN for estimating visual-BMI, age and gender from a face image.
In training, [the IMDB-WIKI dataset, especially wiki_crop version dataset is for age-gender model training, while the BMI dataset is for visual-BMI model training].

###[Log of modification history]

- [Sept. 20, 2020] get to consider and do some research on this topic
- [Oct. 7, 2020] Ready to do the coding [Taking Another Tensorflow-based project](https://github.com/yu4u/age-gender-estimation) as reference


## Dependencies
- Python3.6+， Tensorfow v1.14+, 
-  Other modules in python:
    -- Dlib / MTCNN for running the demo.py

DEMO Tested on:
- MacOS10.15.5 Catalina, Python 3.6.9, Tensorflow 1.14


## Usage of DEMO

### Use trained model for demo
- Run the demo script (requires web cam). You should run the DEMO in the sub-directory of the project path,  
" `test_ABG_estimation` ", 
You can use `--model_func [function_name]` , these arguments indicating the model-function to demo !                             "(bmi-face, age-face, gender-face, abg-face), exclusively.
`--image_dir [IMAGE_DIR]` , optionally to use images in the directory instead. Othersie the WebCam is employed as default. This directory is absolute path.
`--weight_file [MODEL located_DIR]`, path to weight file (e.g. *** weights.***.h5) 

```sh
python demo.py --weight_file trained_model --model_func bmi-face --image_dir /Users/YourName/Pictures/test-ABG-model

```



## Training Age-BMI-Gender model

### Create BMI training data from the BMI-Skymind dataset
Firstly, manually download the raw dataset from the url provided:

Secondly, using the following script to generate transferred  crop-faces dataset version

```sh
python create_db_bmiface_with_margin.py --input original_bmi_dataset --output test_abg_estimation/bmi_face_data

```
Thirdly, using the following script to generate the dataset label auxilary file in MATLAB format, *.mat.

```sh
python creare_db_bmi.py --input bmi_face_data --output bmi_face_data --img_size 224 
```

The resulting files with default parameters are included in this repo (meta/imdb.csv and meta/wiki.csv),
thus there is no need to run this by yourself.


### Train BMI model for the Visual-Face BMI function
Train the model architecture using the training data created above:

```sh
python train_bmi_model.py --bmi_dataset bmi_face_data
```

Trained weight files are stored as `checkpoints/*.h5` 

