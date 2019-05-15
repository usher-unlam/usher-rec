import numpy as np  #instalar con: pip install numpy
import cv2 #instalar con: pip install opencv-python
import numpy as np
import cv2

#cap = cv2.VideoCapture('rtsp://<username_of_camera>:<password_of_camera@<ip_address_of_camera')
cap = cv2.VideoCapture('http://admin:usher@irv.sytes.net:8081')
#con cap = cv2.VideoCapture(0) se captura la web cam de la notebook

while(True):

    ret, frame = cap.read()
    cv2.imshow('Stream IP Camera OpenCV',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()