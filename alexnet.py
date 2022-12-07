from tensorflow.keras.layers import Conv2D  
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers.core import Activation



class AlexNet(object):
    """alexNet model"""
    def __init__(self, num_classes, input_shape):
        self.num_classes = num_classes
        self.input_shape = input_shape
        #build CNN

    def model(self):
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=self.input_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.5))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.5))

        # Output Layer
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        return model

model = AlexNet(2, (224, 224, 3)).model()