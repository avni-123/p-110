# To Capture Frame
import cv2

# To process image array
import numpy as np

# import the tensorflow modules and load the model
import tensorflow as tf

model=tf.keras.models.load_model('keras_model.h5')

# Attaching Cam indexed as 0, with the application software
vid = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the vid 
	status , frame = vid.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
	
		#resize the frame
		resize_img = cv2.resize(frame, (224,224))

		img = np.array(resize_img, dtype = np.float32)
		
		# expand the dimensions
		dimension = np.expand_dims(img, axis=0)
		
		# normalize it before feeding to the model
		normal = dimension/255.0
		
		# get predictions from the model
		prediction = model.predict(normal)
        print('Prediction:', prediction)
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		key = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if key == 32:
			break

# release the vid from the application software
vid.release()

# close the open window
cv2.destroyAllWindows()
