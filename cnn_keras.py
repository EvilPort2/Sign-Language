import numpy as np
import pickle
import cv2
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('tf')

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def cnn_model():
	num_of_classes = 27
	model = Sequential()
	model.add(Conv2D(32, (5,5), input_shape=(image_x, image_y, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(64, (5,5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(num_of_classes, activation='softmax'))
	sgd = optimizers.SGD(lr=1e-4)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	filepath="cnn_model_keras2.h5"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	checkpoint2 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint1, checkpoint2]
	return model, callbacks_list

def train():
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("test_images", "rb") as f:
		test_images = np.array(pickle.load(f))
	with open("test_labels", "rb") as f:
		test_labels = np.array(pickle.load(f), dtype=np.int32)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
	train_labels = np_utils.to_categorical(train_labels)
	test_labels = np_utils.to_categorical(test_labels)

	model, callbacks_list = cnn_model()
	model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=100, batch_size=100, callbacks=callbacks_list)
	scores = model.evaluate(test_images, test_labels, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))
	#model.save('cnn_model_keras2.h5')

train()