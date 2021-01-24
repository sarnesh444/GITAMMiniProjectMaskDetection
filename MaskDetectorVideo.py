import numpy as np
import keras
import cv2
import datetime
from keras.preprocessing import image
from keras.models import load_model

#loading the trained model
mymodel = load_model('mymaskdetectormodel.h5')

#creating object to capture frames in video input
cap = cv2.VideoCapture('CCTVVideoMask.mp4')
#loading face detector classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#reading frames from input video until 'q' is pressed on keyboard
while cap.isOpened():
    _, img = cap.read()
#detecting face in frame
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
#getting face coords
    for (x, y, w, h) in face:
        #processing image
        face_img = img[y:y + h, x:x + w]
        cv2.imwrite('temp.jpg', face_img)
        test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        #predicting target class of image
        pred = mymodel.predict_classes(test_image)[0][0]
        #creating bounding box and adding text
        if pred == 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, 'NO MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        #timestamping the video frame
        datet = str(datetime.datetime.now())
        cv2.putText(img, datet, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    #resizing and displaying individual frames
    imS = cv2.resize(img, (960, 540))
    cv2.imshow('img', imS)

    #loop continues until keyboard interrupt by pressing letter 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()