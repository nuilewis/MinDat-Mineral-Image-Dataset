import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore", category=FutureWarning)

# Preparing Data
IMAGE_SIZE = [224, 224, 3]
BATCH_SIZE = 32
TRAINING_DIR = "downloaded_images/dataset/train"
EPOCHS = 30

# 1. Image Augmentation using an ImageDataGenerator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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

# Loading pretrained model
vgg16 = tf.keras.applications.VGG16(
    input_shape=IMAGE_SIZE, weights='imagenet', include_top=False)

vgg16.trainable = False


# Training our own model

# Adding new layers for our specific task (we could add more convolution layers and dense or pooling etc)
x = tf.keras.layers.Flatten()(vgg16.output)
output_layer = tf.keras.layers.Dense(10, activation='softmax')(x)

# compile the model
model = tf.keras.models.Model(inputs=vgg16.input, outputs=output_layer)

model.summary()
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer= optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])

config = optimizer.get_config()

print("learning rate is ", config['learning_rate'])

# save the best model as we progress
save_best_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model/trained/mineral_identification_model.h5', verbose=2, save_best_only=True)
startingTime = datetime.now()

# Create a CSVLogger object
csv_logger = tf.keras.callbacks.CSVLogger('training_log.csv', append=True, separator=',')

# Training

model_history = model.fit(
    train_generator, validation_data=validation_generator, validation_split=0.2, epochs=EPOCHS, validation_steps=32, callbacks=[save_best_model_callback, csv_logger], verbose=1)



# evaluating
loss0, accuracy0 = model.evaluate(validation_generator)


print("Test loss: {:.2f}".format(loss0))
print("Test accuracy: {:.2f}".format(accuracy0))

model.save('model/trained/mineral_identification_model_train_complete.h5')

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
