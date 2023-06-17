import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


# plt.imshow(training_images[0])
# print(training_images[1])
# print(training_images[1])

#defineing a callback to train our model until it reaches a set accuracy
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.9):
            print("\n Reached 90% accuracy, so cancelling training")
            self.model.stop_training = True

callbacks = myCallback()

#normalising our training sets

training_images = training_images /255
test_images = test_images / 255

#designing the model structure

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation= tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer= tf.keras.optimizers.Adam(),
              loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])

print(test_labels[0])
