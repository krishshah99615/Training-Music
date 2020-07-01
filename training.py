import tensorflow as tf
import matplotlib.pyplot as plt
import random

mnist_data = tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels) = mnist_data.load_data()

print("No of training exapmles "+str(train_images.shape[0]))
print("No of Test exapmles "+str(test_images.shape[0]))
print("Shape of Each Image "+str(train_images[0].shape))

train_images = train_images.reshape(train_images.shape[0],28,28,1)/255
test_images = test_images.reshape(test_images.shape[0],28,28,1)/255

print("Shape of Each Image "+str(train_images[0].shape))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="best_weights.hdf5",
                               monitor = 'val_accuracy',
                               verbose=1,
                               save_best_only=True)
es=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15)
log = tf.keras.callbacks.CSVLogger('model_log.csv',separator=',')
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_images,train_labels,batch_size=128,validation_data=(test_images,test_labels),epochs=100,callbacks=[checkpointer,es,log])