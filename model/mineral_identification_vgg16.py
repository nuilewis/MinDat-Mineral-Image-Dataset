import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore", category=FutureWarning)

# Preparing Data
IMAGE_SIZE = [224, 224, 3]
BATCH_SIZE = 64
TRAINING_DIR = "downloaded_images/dataset/train"
EPOCHS = 50

# 1. Image Augmentation using an ImageDataGenerator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2,
)

# Splitting the Train and Valdiation Sets
train_generator = datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=BATCH_SIZE,
    target_size=(224, 224),
    class_mode='categorical',
    subset='training',
)
validation_generator = datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='validation',
)

test_generator = datagen.flow_from_direcotry(
    TRAINING_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='test'
)

# Loading pretrained model
vgg16 = tf.keras.applications.VGG16(
    input_shape=IMAGE_SIZE, weights='imagenet', include_top=False)

# vgg16.trainable = True

# Freeze the first 15 layers
for layer in vgg16.layers:
    layer.trainable = False


# Training our own model

# Adding new layers for our specific task (we could add more convolution layers and dense or pooling etc)
x = tf.keras.layers.Flatten()(vgg16.output)
output_layer = tf.keras.layers.Dense(10, activation='softmax')(x)

# compile the model
model = tf.keras.models.Model(inputs=vgg16.input, outputs=output_layer)

model.summary()
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])

config = optimizer.get_config()

print("learning rate is ", config['learning_rate'])

# save the best model as we progress
save_best_model = tf.keras.callbacks.ModelCheckpoint(
    filepath='model/trained/mineral_identification_model_best.h5', verbose=2, save_best_only=True)
save_after_epoch = tf.keras.callbacks.ModelCheckpoint(
    filepath='model/trained/mineral_identification_model_after_epoch.h5', verbose=2)
startingTime = datetime.now()

# Create a CSVLogger object
csv_logger = tf.keras.callbacks.CSVLogger(
    'training_log_local.csv', append=True, separator=',')

# Training

model_history = model.fit(
    train_generator, validation_data=validation_generator, validation_split=0.2, epochs=EPOCHS, validation_steps=32, callbacks=[save_best_model, save_after_epoch, csv_logger], verbose=1)


# evaluating
loss0, accuracy0 = model.evaluate(validation_generator)


print("Test loss: {:.2f}".format(loss0))
print("Test accuracy: {:.2f}".format(accuracy0))

model.save('model/trained/mineral_identification_model_train_complete.h5')

#plotting the confusion matrix

predicted_labels = model.predict(validation_generator)
true_labels = validation_generator.classes;

#class_names=["calcite", "copper", "flourite", "galena", "gold", "magenetite", "pyrite", "quartz", "silver", "sphalerite"]

cm = confusion_matrix(true_labels, predicted_labels.argmax(axis=1), normalize = 'pred')


# Plot the confusion matrix using the `matplotlib` library ans seaborn.
plt.figure(figsize=(10,10), dpi = 300)
sns.heatmap(cm, annot=True, cmap="YlGnBu", cbar=False)
plt.xticks(range(10))
plt.yticks(range(10))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# plotting the model training results

acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']


epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


duration = datetime.now()-startingTime
print('Training completed in duration: ', duration)
