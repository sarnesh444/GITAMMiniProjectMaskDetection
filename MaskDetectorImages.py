import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
#Loading the image to be tested
test_image = cv2.imread('34-with-mask.jpg')

#Converting to grayscale
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

#Displaying the grayscale image
plt.imshow(test_image_gray, cmap='gray')
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#loading face detector classifier
haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#detecting faces in image
faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

print('Faces found: ', len(faces_rects))

#loading the trained model
model=load_model('mymaskdetectormodel.h5')

#since the image to be passed to the model has to be preprocessed so loaded again
for (x,y,w,h) in faces_rects:
     tm=image.load_img('34-with-mask.jpg',target_size=(150,150,3))
     tm=image.img_to_array(tm)
     tm=np.expand_dims(tm,axis=0)
#predicting target class using existing model
     pred=model.predict(tm)[0][0]
     print(pred)
#creating bounding box and adding text
     if pred==1:
         cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
         cv2.putText(test_image,'NO MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
     else:
         cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
         cv2.putText(test_image,'MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

#resizing and displaying the image
test_image = cv2.resize(test_image, (300,300))
cv2.imshow('img',test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()