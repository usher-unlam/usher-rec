import numpy as np
import cv2

#cap = cv2.VideoCapture('rtsp://<username_of_camera>:<password_of_camera@<ip_address_of_camera')
cap = cv2.VideoCapture('http://admin:usher@irv.sytes.net:8081')

while(True):

    ret, frame = cap.read()
    cv2.imshow('Stream IP Camera OpenCV',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()