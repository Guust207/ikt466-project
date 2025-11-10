# Import the required libraries
import tensorflow as tf
import keras
from tensorflow.python.client import  device_lib
#from tensorflow import keras
#from tensorflow.keras import layers
from keras import layers
#from tensorflow.keras.models import Sequential
from keras.models import Sequential
#from tensorflow.keras.regularizers import l2
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
import time
from tqdm import tqdm
#from tensorflow.keras import layers, regularizers
from keras import layers, regularizers


# Set random seed for Python's built-in random number generator
import random
random_seed = 42
random.seed(random_seed)

# Set random seed for NumPy
np.random.seed(random_seed)

# Set random seed for TensorFlow
tf.random.set_seed(random_seed)


# Check for configuration of available GPU devices
def cehck_for_avalible_devices():
	return device_lib.list_local_devices()


#preprocessoring of the image then return feature and lables

def pre_process(data_set_dir):

	DATADIR = str(data_set_dir)
	CATEGORIES = ["food_good", "food_bad"]
	dataset = []
	img_size = (224, 224)
	for category in tqdm(CATEGORIES):
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
				image_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
				new_array = cv2.resize(image_rgb, img_size)
				dataset.append([new_array, class_num])
			except Exception as e:
				pass

	X = []  # features
	Y = []  # labels

	for features, label in dataset:
		X.append(features)
		Y.append(label)

	return X, Y


img_size = (224, 224)  # Update this based on your image size


# Split the dataset into training and test sets

def spliting_dataset_trening_valid(X,Y,img_size):


	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

	# Convert to numpy array and reshape into image format
	X_train = np.array(X_train).reshape(-1, img_size[0], img_size[1], 3) #somthing is wrong this img_size is not varible but rather a properti of reshape
	Y_train = np.array(Y_train).reshape(-1, )

	X_test = np.array(X_test).reshape(-1, img_size[0], img_size[1], 3)
	Y_test = np.array(Y_test).reshape(-1, )

	#May require normalization of pixel values
	#X_train = (X_train) / 255
	#X_test = (X_test) / 255
	print("this is picture shape",X_test.shape)

	return X_train, X_test, Y_train, Y_test , img_size






# ---------------------- RESNET BUILDING BLOCKS ----------------------------------------------------
def res_net_block(input_data, filters, conv_size):
	x = layers.Conv2D(filters, conv_size, activation='swish', padding='same')(input_data)
	x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
	x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
	x = layers.Add()([x, input_data])
	x = layers.Activation('swish')(x)

	return x


# Custom Model construction
# -----------------------------------------------------------------------------------------------------
def createModel(img_size):
	# Model construction

	kernelSize = (8, 8)
	maxpoolSize = (6, 6)

	inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
	x = layers.BatchNormalization()(inputs)
	x = layers.Conv2D(16, kernelSize, padding='same', activation='relu6')(x)
	x = layers.MaxPooling2D(pool_size=maxpoolSize, strides=None)(x)

	num_res_net_blocks = 1
	for i in range(num_res_net_blocks):
		x = res_net_block(x, 16, 8)

	x = layers.Conv2D(32, kernelSize, padding='same', activation='relu6')(x)
	x = layers.MaxPooling2D(pool_size=maxpoolSize, strides=None)(x)
	x = layers.Conv2D(64, kernelSize, padding='same', activation='relu6')(x)
	x = layers.MaxPooling2D(pool_size=maxpoolSize, strides=None)(x)
	# x = layers.Dropout(0.3)(x)
	x = layers.Flatten()(x)
	outputs = layers.Dense(1, activation='softmax')(x)
	model = tf.keras.Model(inputs, outputs)
	model.summary()
	return model


# Early stopping

# Create the model
def creating_model(img_size):
	model = createModel(img_size)
	return model


# Model compile
def compling_mdel(X_train, Y_train,The_created_model):
	model = The_created_model
	model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

	# Model fit/train
	history = model.fit(X_train, Y_train, epochs=40, shuffle=True, batch_size=64, validation_split=0.1)

	return history


# Accuracy curves
def acc_plt(history):
	plt.plot(history.history['accuracy'], label='train_accuracy')
	plt.plot(history.history['val_accuracy'], label='validation_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0, 1])
	plt.legend(loc='lower right')
	plt.savefig('', facecolor='white')
	plt.show()


# Loss curves
def loss_plt(history):
	plt.plot(history.history['loss'], label='train_loss')
	plt.plot(history.history['val_loss'], label='validation_loss')
	plt.xlabel('Epoch')
	plt.ylabel('loss')
	plt.ylim([0, 1])
	plt.legend(loc='upper right')
	plt.savefig('', facecolor='white')
	plt.show()


# Calculate the desired metrics

def needed_metrics(X_test, Y_test, The_created_model):
	model = The_created_model
	test_loss, test_acc = model.evaluate(X_test, Y_test)

	# Model prediction provides probabilities of the input class
	# Based on the probabilities the corresponding class category is determined.
	Y_te = np.array(tf.math.argmax(model.predict(X_test), 1))

	# Calculate the accuracy metrics
	acc = metrics.accuracy_score(Y_test, Y_te)

	# Classification metrics and report
	classReport = classification_report(Y_test, Y_te)

	# Print and save all the metrics
	print("test_accuracy:", acc * 100, "\n")
	print("test_loss:", test_loss, "\n")
	print(classReport)

	return Y_te


# Plot the confusion matrix
def confusion_matrix(Y_test, Y_te):


	con_mat = tf.math.confusion_matrix(labels=Y_test, predictions=Y_te).numpy()
	con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
	classes = ["Rain", "Ocean Waves", "Burning Fire", "Bee", "Bird", "Wind", "Pour water", "Water flushing",
			   "Thunderstorm"]
	con_mat_df = pd.DataFrame(con_mat_norm,
							  index=classes,
							  columns=classes)
	figure = plt.figure(figsize=(7, 5))
	sns.heatmap(con_mat_df, annot=True, cmap="Blues")
	plt.tight_layout()
	# plt.title('Convolution Neural Newtork')
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	# plt.savefig(resultsStore+modelName+'/confusionMatrix.png',facecolor='white')
	plt.show()


def model_summary(The_created_model):
	model = The_created_model
	model.summary()


