#Author: Rhushil Vasavada
#Devanagari Character CNN Tensorflow Model
#Description: This is a custom-built Convolutional Neural Network (CNN) that 
#has been trained on 9,000+ handwritten Devanagari characters. It contains several
#layers such as max pooling and flattening to perform matrix transformations 
#on the input matrix representing a 2D array of the handwritten character (in grayscale).

#import libraries
import tensorflow as tf
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import EarlyStopping

#create a generator to be used to manipulate training data
trainDataGen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest')

#create a generator to be used to manipulate testing data
test_datagen = ImageDataGenerator(rescale=1. / 255)

#load in training data 
train_generator = trainDataGen.flow_from_directory(
    "DevanagariHandwrittenCharacterDataset/Train",
    target_size=(32, 32),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical")
prev = ""

#these are the hindi characters that make up the training data (in english forms)
labels = ["ka", "kha", "ga", "gha", "kna", "cha", "chha", "ja", "jha", "yna", "t`a", "t`ha", "d`a", "d`ha", "adna",
          "ta", "tha", "da", "dha", "na", "pa", "pha", "ba", "bha", "ma", "yaw", "ra", "la", "waw", "sha", "shat", "sa",
          "ha", "aksha", "tra", "gya", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
count = 0;

#load in testing data
validation_generator = test_datagen.flow_from_directory(
    "DevanagariHandwrittenCharacterDataset/Test",
    target_size=(32, 32),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical')

#build model (with TensorFLow sequential framework to add several layers) 
model = Sequential()

#add convolutional layers
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=1,
                 padding='same',
                 activation="relu",
                 input_shape=(32, 32, 1)))

model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=1,
                 padding='same',
                 activation="relu",
                 input_shape=(32, 32, 1)))

#add max pooling layers
model.add(MaxPool2D(pool_size=(2, 2),
                    strides=(2, 2),
                    padding="same"))

#more convolutional layers
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding='same',
                 strides=1,
                 activation="relu"))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=1,
                 padding='same',
                 activation="relu"))

#max pooling
model.add(MaxPool2D(pool_size=(2, 2),
                    strides=(2, 2),
                    padding="same"))

#more layers
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128,
                activation="relu",
                kernel_initializer="uniform"))

model.add(Dense(64,
                activation="relu",
                kernel_initializer="uniform"))

model.add(Dense(46,
                activation="softmax",
                kernel_initializer="uniform"))

#finally, we are using adam optmizer for gradient descent 
#and other parameters to compile model
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary())

early_stopping_monitor = EarlyStopping(patience=2)

#this essentially fits the model on the training data, with 10 epochs and specifies other parameters to enhance model performance
with tf.device('/gpu:0'):
    history = model.fit_generator(train_generator, epochs=10, validation_data=validation_generator,
                                  steps_per_epoch=2444, validation_steps=432, use_multiprocessing=True,
                                  callbacks=[early_stopping_monitor])

#save model as pickle file to be used in scratchpad application
model.save("hindi_OCR_cnn_model_tf2.h5")
pickle.dump(model, open('hindi_OCR_cnn_model_pickle.pkl2', 'wb'))
