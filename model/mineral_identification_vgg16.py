import os
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import preprocess_input
from keras_preprocessing.image import ImageDataGenerator
from datetime import datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Preparing Data
IMAGE_SIZE = [224, 224, 3]

# Loading pretrained model
vgg16 = tf.keras.applications.VGG16(
    input_shape=IMAGE_SIZE, weights='imagenet', include_top=False)

vgg16.trainable = False


TRAINING_DIR = "downloaded_images/dataset/train"
#VALIDATION_DIR = "downloaded_images/dataset/test"



# 1. Image Augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# val_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rescale=1./255,
# )

# resizing images to fit the model
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
)

# validation_generator = val_datagen.flow_from_directory(
#     VALIDATION_DIR,
#     target_size=(224, 224),
#     class_mode='categorical',
#     batch_size=32,
# )


# get number of classes from the number of folders

# Training our ouwn model
flattenedInput = Flatten()(vgg16.output)
prediction = Dense(10, activation='softmax')(flattenedInput)
model = tf.keras.Model(inputs=vgg16.input, outputs=prediction)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy', metrics=['accuracy'])

# save the best model as we progress
callbacks = tf.keras.callbacks.ModelCheckpoint(
    filepath='model/trained/mymodel.h5', verbose=2, save_best_only=True)
startingTime = datetime.now()

# Training
try:
  model_history = model.fit(
      train_generator, validation_split=.2, epochs=5, validation_steps=32, callbacks=callbacks, verbose=1,)
except Exception as e:
  print(f'Error: {e}')
  
#evaluating
loss0, accuracy0 = model.evaluate()
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

## plotting the model training results

acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']


epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


duration = datetime.now()-startingTime
print('Training completed in duration: ', duration)