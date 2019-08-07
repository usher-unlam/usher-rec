import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import collections
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
from datetime import datetime as time
from urllib.parse import urlparse

import glob
import xml.etree.ElementTree as ET

from collections import namedtuple


import socket
def urlTest(host, port):
    out = (0,"")
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

#A partir de un xml previamente cargado con labelImage, obtengo la posicion de cada ubicacion
#(correspondiente a cada banca) dentro de la toma de video completa
def xml_to_locations(path):
    locations_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (int(member[4][0].text), #xmin
                     int(member[4][1].text), #ymin
                     int(member[4][2].text), #xmax
                     int(member[4][3].text)  #ymax
                     )
            locations_list.append(value)
    return locations_list

#Area de interseccion entre 2 rectangulos
#Para determinar coincidencia entre las posiciones del xml y las boxes de la CNN
def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy

#ipcamUrl = 0
#ipcamUrl = 'http://admin:usher@irv.sytes.net:8081'
ipcamUrl = 'http://admin:usher@192.168.1.34:8081'
ipcam = {}
ipcamDesc = 'Celular'
ipcam[ipcamDesc] = urlparse(ipcamUrl)
print(time.now())

# Prueba la conexión al destino ip
if len(ipcamUrl) > 5:
  err,errMsg = urlTest(ipcam[ipcamDesc].hostname,ipcam[ipcamDesc].port)
  if err > 0:
      print(time.now(),"Falló conexión. ",errMsg)
      exit(1)

try:
  cap = cv2.VideoCapture(ipcamUrl)

  sys.path.append("..")

  # Importación del módulo de detección de objetos.
  from object_detection.utils import label_map_util
  from object_detection.utils import visualization_utils as vis_util

  PATH_TO_CKPT = 'modelo_congelado/frozen_inference_graph.pb'

  PATH_TO_LABELS = os.path.join('configuracion', 'label_map.pbtxt')

  NUM_CLASSES = 90
  FRAMES_OMITIDOS = 10 #Análisis en LAN: frames{fluido,delay}= 4{si,>4"} 7{si,<1"} 10{si,~0"}

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')


  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  
  #Obtengo la posicion, dentro de la toma completa, de cada ubicacion 
  path_locations='configuracion'
  images_location=xml_to_locations(path_locations)
  
 
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      sess = tf.Session()
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      i = FRAMES_OMITIDOS
      while True:
        ret, image_np = cap.read()
        if (i < FRAMES_OMITIDOS):
          i += 1
        else:
          i = 0    
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.      
          (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})

          box = np.squeeze(boxes)
          #Alto del frame en pixeles
          height = np.size(image_np, 0)
          #Ancho del frame en pixeles
          width = np.size(image_np, 1)
         
          ##Comparo cada rectangulo del xml con cada box de la CNN
          ##Si el porcentaje de coincidencia es mayor a PORC_INTERSECCION guardo "[OK] "
          ##Si no, guardo "[ ] "
          locations_state=""
          personas=0
          Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
          PORC_INTERSECCION=0.5
          
          #Recorro las posiciones del xml
          for j in images_location:
            ymin = int(j[1])
            xmin = int(j[0])
            ymax = int(j[3])
            xmax = int(j[2])
            area_xml=(ymax-ymin)*(xmax-xmin)
            rxml = Rectangle(xmin, ymin, xmax, ymax)
            #Para cada posicion recorro las boxes buscando coincidencia
            coincide=False
            for index,value in enumerate(classes[0]):
             if scores[0,index] > 0.3:
               if category_index.get(value).get('name')=="person":
                 ymin = (int(box[index,0]*height))
                 xmin = (int(box[index,1]*width))
                 ymax = (int(box[index,2]*height))
                 xmax = (int(box[index,3]*width))
                 rbox = Rectangle(xmin, ymin, xmax, ymax)
                 area_interseccion=area(rxml, rbox)
                 if(area_interseccion!=None):
                   if area_interseccion>(PORC_INTERSECCION*area_xml):
                     coincide=True     
                  
            if coincide==True:
              locations_state=locations_state+"[OK] "
              personas+=1
            else:
              locations_state=locations_state+"[ ] "

          print ("Se detectaron "+str(personas)+" personas\n")
          print (locations_state+"\n\n")

          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
                    
          cv2.imshow('object detection', image_np)
          if cv2.waitKey(25) & 0xFF == ord('q'):
            break
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

