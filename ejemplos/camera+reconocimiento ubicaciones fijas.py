import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import winsound
frequency = 1500  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second

import glob
import xml.etree.ElementTree as ET
import collections
from collections import defaultdict
from io import StringIO
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

import cv2
cap = cv2.VideoCapture('http://admin:admin@192.168.1.33:8081')

sys.path.append("..")

# Importación del módulo de detección de objetos.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_CKPT = 'modelo_congelado/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('configuracion', 'label_map.pbtxt')

NUM_CLASSES = 90

#A partir de un xml previamente cargado con labelImage, obtengo la posicion de cada ubicacion
#(correspondiente a cada banca) dentro de la toma de video completa
def xml_to_locations(path):
    locations_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (int(member[4][1].text), #ymin
                     int(member[4][3].text)-int(member[4][1].text), #height
                     int(member[4][0].text), #xmin
                     int(member[4][2].text)-int(member[4][0].text) #width
                     )
            locations_list.append(value)
    return locations_list

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
min_score_thresh = 0.5

IMAGE_SIZE = (12, 8)

ret, image_np = cap.read()
cont=0
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
    #(left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    while True:
           
      if cont==5:
       cont=0
       personas=0
       locations=""
       ret, image_np = cap.read()   
       #Recorto la imagen segun las posiciones fijas
       for j in images_location:
         y=j[0]
         h=j[1]
         x=j[2]
         w=j[3]
         crop_img = image_np[y:y+h, x:x+w]
         # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
         #image_np_expanded = np.expand_dims(image_np, axis=0)
         image_np_expanded = np.expand_dims(crop_img, axis=0)
  
         # Actual detection.      
         (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],  feed_dict={image_tensor: image_np_expanded})
  
         # Visualization of the results of a detection.
         vis_util.visualize_boxes_and_labels_on_image_array(
         crop_img,
         np.squeeze(boxes),
         np.squeeze(classes).astype(np.int32),
         np.squeeze(scores),
         category_index,
         use_normalized_coordinates=True,
         line_thickness=8)
       
         #Personas detectadas con un 50% de certeza
         hay_persona=False
         for index,value in enumerate(classes[0]):
           if scores[0,index] > 0.3:
             if category_index.get(value).get('name')=="person":
               hay_persona=True
               #print (category_index.get(value))

         if hay_persona==True:
           personas+=1
           locations=locations+"[OK] "
         else:
           locations=locations+"[ ] "
        
       print ("Se detectaron "+str(personas)+" personas\n")
       print (locations+"\n\n")
       
       #print(scores)
       #Alto del frame en pixeles
       #height = np.size(image_np, 0)
       #Ancho del frame en pixeles
       #width = np.size(image_np, 1)
       #cant=int(num)
       #print ("Se detectaron "+str(cant)+" personas")
       #box = np.squeeze(boxes)
       #for i in range(personas):
         #if classes[i] in category_index.keys():
         #class_name = category_index.get(1).get('name')
         
         #else:
         #class_name = 'N/A'
         #ymin = (int(box[i,0]*height))
         #xmin = (int(box[i,1]*width))
         #ymax = (int(box[i,2]*height))
         #xmax = (int(box[i,3]*width))
         #print(class_name,xmin,xmax,ymin,ymax,classes[0])
                   
      else:
       cont=cont+1
      

      
      cv2.imshow('object detection', image_np)
      if cv2.waitKey(33) == ord('a'):
        winsound.Beep(frequency, duration)
        

      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
