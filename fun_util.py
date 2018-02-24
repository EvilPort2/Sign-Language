import cv2, pickle
import numpy as np
import tensorflow as tf
from cnn_tf import cnn_model_fn
import os
import sqlite3
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras2.h5')

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def recognize():
	global prediction
	cam = cv2.VideoCapture(1)
	hist = get_hand_hist()
	x, y, w, h = 300, 100, 300, 300
	text = ""
	word = ""
	count_frame = 0
	while True:
		img = cam.read()[1]
		img = cv2.flip(img, 1)
		imgCrop = img[y:y+h, x:x+w]
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		cv2.filter2D(dst,-1,disc,dst)
		blur = cv2.GaussianBlur(dst, (11,11), 0)
		blur = cv2.medianBlur(blur, 15)
		thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		thresh = cv2.merge((thresh,thresh,thresh))
		thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
		thresh = thresh[y:y+h, x:x+w]
		contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

		old_text = text
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			#print(cv2.contourArea(contour))
			if cv2.contourArea(contour) > 10000:
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				save_img = thresh[y1:y1+h1, x1:x1+w1]
				
				if w1 > h1:
					save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif h1 > w1:
					save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
				pred_probab, pred_class = keras_predict(model, save_img)
				print(pred_class, pred_probab)
				if pred_probab*100 > 70:
					text = get_pred_text_from_db(pred_class)

				if old_text == text:
					count_frame += 1
				else:
					count_frame = 0

				if count_frame == 20:
					word = word + text
			else:
				text = ""
				word = ""
		else:
			text = ""
			word = ""
		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		#splitted_text = split_sentence(text, 2)
		#put_splitted_text_in_blackboard(blackboard, splitted_text)
		cv2.putText(blackboard, text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255))
		cv2.putText(blackboard, word, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((img, blackboard))
		cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)
		if cv2.waitKey(1) == ord('q'):
			break

		
recognize()