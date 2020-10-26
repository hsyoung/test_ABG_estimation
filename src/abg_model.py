from tensorflow.python.keras import Model

from tensorflow.python.keras import models
from tensorflow.python.keras import layers

def get_model(model_name="VGG-face"):
    # vgg_model = None

    # VGG-Face model
    vgg_model = models.Sequential()
    vgg_model.add(layers.ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    vgg_model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    vgg_model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
    vgg_model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
    vgg_model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(layers.ZeroPadding2D((1, 1)))
    vgg_model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(layers.Convolution2D(4096, (7, 7), activation='relu'))
    vgg_model.add(layers.Dropout(0.5))
    vgg_model.add(layers.Convolution2D(4096, (1, 1), activation='relu'))
    vgg_model.add(layers.Dropout(0.5))
    vgg_model.add(layers.Convolution2D(2622, (1, 1)))
    vgg_model.add(layers.Flatten())
    vgg_model.add(layers.Activation('softmax'))

    if model_name == "VGG-face":
        model = Model(inputs=vgg_model.input, outputs=vgg_model.output)
    elif model_name == "age-face":
        base_age_model = models.Sequential()
        base_age_model = layers.Convolution2D(101, (1, 1), name='predictions_age')(vgg_model.layers[-4].output)
        base_age_model = layers.Flatten()(base_age_model)
        base_age_model = layers.Activation('softmax')(base_age_model)

        model = Model(inputs=vgg_model.input, outputs=base_age_model)
    elif model_name == "gender-face":
        base_gender_model = models.Sequential()
        base_gender_model = layers.Convolution2D(2, (1, 1), name='predictions_gender')(vgg_model.layers[-4].output)
        base_gender_model = layers.Flatten()(base_gender_model)
        base_gender_model = layers.Activation('softmax')(base_gender_model)

        model = Model(inputs=vgg_model.input, outputs=base_gender_model)
    elif model_name == "bmi-face":
        # Scheme 1 of bmi_model

        base_bmi_model = models.Sequential()
        base_bmi_model = layers.Convolution2D(160, (1, 1))(vgg_model.layers[-4].output)
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
        base_bmi_model = layers.Convolution2D(101, (1, 1), name='ref_agemodel')(vgg_model.layers[-4].output)
        base_bmi_model = layers.advanced_activations.ReLU()(base_bmi_model)
        # base_bmi_model = layers.Convolution2D(1, (1, 1), name='predictions_bmi')(base_bmi_model)
        base_bmi_model = layers.Flatten()(base_bmi_model)
        base_bmi_model = layers.Dense(1, kernel_initializer='normal', name='predictions_bmi')(base_bmi_model)
        """
        # Scheme 3 of bmi_model
        """
        base_bmi_model = models.Sequential()
        base_bmi_model = layers.Convolution2D(101, (1, 1), name='ref_agemodel')(vgg_model.layers[-4].output)
        base_bmi_model = layers.advanced_activations.ReLU()(base_bmi_model)
        base_bmi_model = layers.Convolution2D(1, (1, 1), name='predictions_bmi')(base_bmi_model)
        base_bmi_model = layers.Flatten()(base_bmi_model)
        """
        model = Model(inputs=vgg_model.input, outputs=base_bmi_model)

    elif model_name == "abg-face":
        base_age_model = models.Sequential()
        base_age_model = layers.Convolution2D(101, (1, 1), name='predictions_age')(vgg_model.layers[-4].output)
        base_age_model = layers.Flatten()(base_age_model)
        base_age_model = layers.Activation('softmax')(base_age_model)

        base_gender_model = models.Sequential()
        base_gender_model = layers.Convolution2D(2, (1, 1), name='predictions_gender')(vgg_model.layers[-4].output)
        base_gender_model = layers.Flatten()(base_gender_model)
        base_gender_model = layers.Activation('softmax')(base_gender_model)

        base_bmi_model = models.Sequential()
        base_bmi_model = layers.Convolution2D(160, (1, 1))(vgg_model.layers[-4].output)
        base_bmi_model = layers.advanced_activations.PReLU()(base_bmi_model)
        base_bmi_model = layers.Dropout(0.4)(base_bmi_model)
        base_bmi_model = layers.Convolution2D(32, (1, 1))(base_bmi_model)
        base_bmi_model = layers.advanced_activations.PReLU()(base_bmi_model)
        base_bmi_model = layers.Dropout(0.3)(base_bmi_model)
        base_bmi_model = layers.Convolution2D(1, (1, 1), name='predictions_bmi')(base_bmi_model)
        base_bmi_model = layers.Dropout(0.2)(base_bmi_model)
        base_bmi_model = layers.Flatten()(base_bmi_model)

        model = Model(inputs=vgg_model.input, outputs=[base_gender_model,
                                                       base_age_model,
                                                       base_bmi_model])
    else:
        print("Error:\n"
              "the request model name is not supported by the program...\n ")
        return

    return model


def main():
    model = get_model("bmi-face")
    print(" This default running displays the scheme of Visual-BMI model: \n ")
    model.summary()


if __name__ == '__main__':
    main()
