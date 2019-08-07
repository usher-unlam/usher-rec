import sys
import socket
import argparse

def urlTest(host, port):
    out = (0,"")
    # setup argument parsing
    # parser = argparse.ArgumentParser(description='Socket Error Examples')
    # parser.add_argument('--host', action="store", dest="host", required=False)
    # parser.add_argument('--port', action="store", dest="port", type=int, required=False)
    # parser.add_argument('--file', action="store", dest="file", required=False)
    # given_args = parser.parse_args()
    # host = given_args.host
    # port = given_args.port
    # filename = given_args.file
    
    # First try-except block -- create socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.6)
    except socket.error as e:
        out = (1, "Error creating socket: %s" % e)
    # Second try-except block -- connect to given host/port
    else:
        try:
            s.connect((host, port))
        except socket.gaierror as e:
            out = (2, "Address-related error connecting to server: %s" % e)
        except socket.error as e:
            out = (3, "Connection error: %s" % e)
        finally:
            s.close()
    return out
    # # Third try-except block -- sending data
    # try:
    #     s.sendall("GET %s HTTP/1.0\r\n\r\n" % filename)
    # except socket.error as e:
    #     out = (3, "Error sending data: %s" % e)
    # while 1:
    #     # Fourth tr-except block -- waiting to receive data from remote host
    #     try:
    #         buf = s.recv(2048)
    #     except socket.error as e:
    #         out = (4, "Error receiving data: %s" % e)
    #     if not len(buf):
    #         break
    #     # write the received data
    #     sys.stdout.write(buf)


import numpy as np  #instalar con: pip install numpy
import cv2 #instalar con: pip install opencv-python

from datetime import datetime as time
from urllib.parse import urlparse

ipcam = {}
ipcamDesc = 'Celular'
ipcamUrl = 'http://admin:usher@irv.sytes.net:8081'
ipcam[ipcamDesc] = urlparse(ipcamUrl)
print(ipcam[ipcamDesc].password)
#ipcamUrl = 0 # se captura la web cam de la notebook
#cap = cv2.VideoCapture('rtsp://<username_of_camera>:<password_of_camera@<ip_address_of_camera')
print(time.now())

# Prueba la conexión al destino ip
err,errMsg = urlTest(ipcam[ipcamDesc].hostname,ipcam[ipcamDesc].port)
if err > 0:
    print(time.now(),"Falló conexión. ",errMsg)
    exit(1)
try:
    #cap = cv2.VideoCapture(ipcamUrl)
    cap = cv2.VideoCapture()
    print(time.now(),"cap video capture")
    cap.open(ipcamUrl)
    print(time.now(),"cap video opening")
    if cap.isOpened():
        print(time.now(), "cap is Opened")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            cv2.imshow('Stream IP Camera OpenCV',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0: 
        #     faces = face_cascade.detectMultiScale(gray,
        #                                         scaleFactor=1.5,
        #                                         minNeighbors=5,
        #                                         minSize=(30, 30))
        #     for (x, y, w, h) in faces:
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
        #         cv2.imwrite('1/frames/%sf%s.jpg'%(now, str(cap.get(cv2.CAP_PROP_POS_FRAMES))), frame)
    if not cap.isOpened():
        print(time.now(), "No se recibe stream de origen: ", ipcamUrl )
except IOError as e:
    print(time.now(), "Error abriendo socket: ", ipcamUrl)
except KeyboardInterrupt as e:
    print(time.now(), "Detenido por teclado.")
except BaseException as e:
    print(time.now(), "Error desconocido: ", e)
#    if e.number == -138:
#        print("Compruebe la conexión con '" + ipcamUrl + "'")
#    else:
#        print("Error: " + e.message)
finally:
    cap.release()
    cv2.destroyAllWindows()

'''
def multi_camera():
    mirror=False
    #Setting up cameras
    cam0 = cv2.VideoCapture(0)
    cam1 = cv2.VideoCapture(1)
    cam2 = cv2.VideoCapture(2)
    cam3 = cv2.VideoCapture(3)
    cam4 = cv2.VideoCapture(4)
    while True:
        #Getting imgs from cameras 
        ret_val0, img0 = cam0.read()
        ret_val1, img1 = cam1.read()
        ret_val2, img2 = cam2.read()
        ret_val3, img3 = cam3.read()
        ret_val4, img4 = cam4.read()

        cv2.imshow('webcam0',img0)
        cv2.imshow('webcam1',img1)
        cv2.imshow('webcam2',img2)
        cv2.imshow('webcam3',img3)
        cv2.imshow('webcam4',img4)
        #quit
        if cv2.waitKey(1) & 0XFF==ord('q'):
            break
    cv2.destroyAllWindows()'''