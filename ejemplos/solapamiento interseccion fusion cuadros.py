import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

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

def compute_inters(box, boxes, box_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    return np.divide(np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0),box_area)
def compute_intersection(boxes1, boxes2):
    # Area of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    # Compute intersection to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the Intersection value.
    intersection = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(intersection.shape[1]):
        box2 = boxes2[i]
        intersection[:, i] = compute_inters(box2, boxes1, area1)
    return intersection
    
#Intersection Over Union (intersection / union) available in mrcnn.utils.compute_overlaps()
def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def fusion(a, b):  
    # espera (ymin,xmin,ymax,xmax) para a y b
    return (min(a[0], b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3]))



# #ipcamUrl = 0
# #ipcamUrl = 'http://admin:usher@irv.sytes.net:8081'
# ipcamUrl = 'http://admin:usher@192.168.0.9:8081'
# #ipcamUrl = 'rtsp://admin:usher@192.168.0.9:8554/live'
# ipcam = {}
# ipcamDesc = 'Celular'
# ipcam[ipcamDesc] = urlparse(ipcamUrl)
# print(time.now())

# # Prueba la conexión al destino ip
# if len(ipcamUrl) > 5:
#   err,errMsg = urlTest(ipcam[ipcamDesc].hostname,ipcam[ipcamDesc].port)
#   if err > 0:
#       print(time.now(),"Falló conexión. ",errMsg)
#       exit(1)

# try:
#   cap = cv2.VideoCapture(ipcamUrl)
   
#   sys.path.append("..")

#   # Importación del módulo de detección de objetos.
#   from object_detection.utils import label_map_util
#   from object_detection.utils import visualization_utils as vis_util

#   PATH_TO_CKPT = 'modelo_congelado/frozen_inference_graph.pb'

#   PATH_TO_LABELS = os.path.join('configuracion', 'label_map.pbtxt')

#   NUM_CLASSES = 90
#   FRAMES_OMITIDOS = 10 #Análisis en LAN: frames{fluido,delay}= 4{si,>4"} 7{si,<1"} 10{si,~0"}

#   detection_graph = tf.Graph()
#   with detection_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#       serialized_graph = fid.read()
#       od_graph_def.ParseFromString(serialized_graph)
#       tf.import_graph_def(od_graph_def, name='')


#   label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#   categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#   category_index = label_map_util.create_category_index(categories)

#   PATH_TO_TEST_IMAGES_DIR = 'img_pruebas'
#   TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

#   IMAGE_SIZE = (12, 8)

#   contador=0
#   with detection_graph.as_default():
#     with tf.Session(graph=detection_graph) as sess:
#       sess = tf.Session()
#       image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#       # Each box represents a part of the image where a particular object was detected.
#       detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#       # Each score represent how level of confidence for each of the objects.
#       # Score is shown on the result image, together with the class label.
#       detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#       detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#       print("Detection Classes:",detection_classes)
#       num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#       i = FRAMES_OMITIDOS
#       while True:
#         if not cap.isOpened():
#           print(time.now(), "No se recibe stream de origen: ", ipcamUrl )
#           break
#         ret, image_np = cap.read()
#         if ret:
#           if (i < FRAMES_OMITIDOS):
#             i += 1
#           else:
#             i = 0    
#             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#             image_np_expanded = np.expand_dims(image_np, axis=0)
#             # Actual detection.      
#             ##(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
#             # Actual detection.      
#             (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})


#             box = np.squeeze(boxes)
#             #Alto del frame en pixeles
#             height = np.size(image_np, 0)
#             #Ancho del frame en pixeles
#             width = np.size(image_np, 1)
            
#             Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

#             wmin,wmax = (0,0)
#             sillas = []
#             scentros = []
#             sdims = []
#             for index,value in enumerate(classes[0]):
#               if scores[0,index] > 0.3:
#                 if category_index.get(value).get('name')=="chair":
#                   print("Class Value for chair:", value, box[index])
#                   ymin = (int(box[index,0]*height))
#                   xmin = (int(box[index,1]*width))
#                   ymax = (int(box[index,2]*height))
#                   xmax = (int(box[index,3]*width))
#                   rbox = Rectangle(xmin, ymin, xmax, ymax)
#                   #operado como (Y,X) a diferencia de Rectangle
#                   sillas.append( (box[index] * [height,width,height,width]).astype(int))
#                   #Prueba con (Ycentro,Xcentro) y (Alto,Ancho)
#                   cbox = (int((ymin+ymax)/2),int((xmin+xmax)/2)) #centro:(Y,X)
#                   scentros.append(cbox)
#                   dbox = (ymax-ymin,xmax-xmin)
#                   sdims.append(dbox) #dimension: (alto,ancho)
#                   #print('Silla ',index,' [Xmin,Ymin,Xmax,Ymax]=',rbox,' Centro=',cbox,' Dimensiones:',dbox)

s = [[163, 279, 441, 484], [143, 292, 289, 482], [ 87, 201, 295, 338], [108, 323, 276, 500], [107, 198, 252, 305]]
c = [(302, 381), (216, 387), (191, 269), (192, 411), (179, 251)]
d = [(278, 205), (146, 190), (208, 137), (168, 177), (145, 107)]
dmin = [[145, 107]]
#print(s,c,d)
sillas = np.array(s)
scentros = np.trunc(np.concatenate((np.expand_dims(np.divide(sillas[:, 0] + sillas[:, 2],2),axis=1) , np.expand_dims(np.divide(sillas[:, 1] + sillas[:, 3],2),axis=1)),axis=1))
sdims = np.concatenate((np.expand_dims(sillas[:, 2] - sillas[:, 0],axis=1) , np.expand_dims(sillas[:, 3] - sillas[:, 1],axis=1)),axis=1).astype('int32')
#print(sillas,scentros,sdims)
x =  [[1.       , 0.39381477,0.10023424,0.26546335,0.03296719],
     [0.39381477, 1.        , 0.13562197, 0.58209695, 0.03386873],
     [0.10023424, 0.13562197, 1.        , 0.04523262, 0.52124019],
     [0.26546335, 0.58209695, 0.04523262, 1.        , 0.        ],
     [0.03296719, 0.03386873, 0.52124019, 0.        , 1.        ]]
id =  [[2., 3.],
 [1., 3.],
 [1., 2.],
 [1., 3.],
 [1., 2.]]
if True:          
            if len(sillas) > 0:
              print('Sillas:\n',sillas)
              print('Centros:\n',scentros)
              print('Dimensiones:\n',sdims)
              
              #Obtener dimensión mínima de sillas 
              #TODO: ALMACENAR JUNTO CON UBICACIONES DE SILLAS
              sdimmin = np.amin(sdims, axis=0).reshape(1,2)
              print('Dimensión Mínima:\n',sdimmin)
              ##sdminrect = np.concatenate(([[0,0]],sdimmin),axis=1)
              ##print(sdminrect)
              
              #Calcular solapamiento de sillas
              #TODO: fusionar rectángulos solapados de sillas en uno mayor
              sillas = np.array(sillas)
              soverlap = compute_overlaps(sillas,sillas)
              print('Solapamiento Sillas:\n', soverlap)
              #Calcular intersección entre sillas
              sinters = compute_intersection(sillas,sillas)
              print('Intersección Sillas:\n', sinters)

              #Dividir centros de rectángulos con dimensión mínima 
              # para obtener una identificación unívoca
              sident = np.trunc(np.true_divide(scentros,sdimmin))
              print(sident.shape,sillas.shape)
              print('Identificación de Sillas (DimMin Original ',sdimmin[0],'):\n Silla > Centro -> Coordenada')
              for p in range(sident.shape[0]):
                  print(sillas[p],' > ',scentros[p],'->',sident[p])
              #print('Identificación de Sillas:\n',np.concatenate((sident,sillas),axis=1))
              #Remover identificaciones duplicadas
              sident,sunqIDX,sdupIDX,sdupCOUNT = np.unique(sident, return_index=True, return_inverse=True, return_counts=True, axis=0)
              print('Identificación de Sillas (no duplicados):\n',sident)
              #print('unqIDX',sunqIDX,'\ndupIDX',sdupIDX,'\ndupCOUNT',sdupCOUNT)
#d1,d2,d3,d4 = np.unique(id, return_index=True, return_inverse=True, return_counts=True, axis=0)
#print(np.array(id).astype('int32'),'\n',d2,d3,d4,'\n',d1.astype('int32'))
              #fusionar rectángulos id duplicados
              f = np.zeros((sident.shape[0],sillas.shape[1]))
              for i,idup in enumerate(sdupIDX):
                  #si no está agregado, lo agrego en la posición idup
                  if np.sum(f[idup]) == 0:
                    f[idup] = sillas[sunqIDX[idup]]
                  else:
                    f[idup] = fusion(f[idup],sillas[sunqIDX[idup]])
              print('Sillas duplicadas fusionadas:\n',f)
              
              print('------------- PRUEBA DE FUSION / EXTENSIÓN --------------')
              #Fusionar/Extender rectángulos solapados
              solapX = np.array(sillas).shape[0]
              sillasf = []
              #print('Inicio de Fusión/Extensión')
              for i in range(solapX - 1):
                  if sinters[i][i] == 1:
                      ins = s[i]
                      for j in range(i+1, np.array(x).shape[0]):
                        if sinters[j][j] != 0:
                          #Empleando SOLAPAMIENTO
                          if soverlap[i][j] > 0.5:
                              ins = fusion(ins,s[j])
                              sinters[j][j] = 0 #anulo el posterior procesamiento de este elemento
                              print(i+1,',',j+1,'  ', ins, ' x solapamiento')
                          #Empleando INTERSECCIÓN
                          elif max(sinters[i][j],sinters[j][i]) > 0.7:
                              ins = fusion(ins,s[j])
                              sinters[j][j] = 0 #anulo el posterior procesamiento de este elemento
                              print(i+1,',',j+1,'  ', ins, ' x intersección')
                      sillasf.append(ins)
              sillasf = np.array(sillasf)
              print('Sillas Fusionadas:\n',sillasf)
              #PRUEBA DE FUSION/EXTENSION CONTRA SILLAS
              #print('Solapamiento Fusión en Silla:\n', compute_overlaps(sillasf,sillas))
              #print('Intersección Fusión en Silla:\n', compute_intersection(sillasf,sillas))
              #print('Solapamiento Silla en Fusión:\n', compute_overlaps(sillas,sillasf).T)
              print('Intersección Silla en Fusión:\n', compute_intersection(sillas,sillasf).T)
              
              sfcentros = np.trunc(np.concatenate((np.expand_dims(np.divide(sillasf[:, 0] + sillasf[:, 2],2),axis=1) , np.expand_dims(np.divide(sillasf[:, 1] + sillasf[:, 3],2),axis=1)),axis=1))
              sfdims = np.concatenate((np.expand_dims(sillasf[:, 2] - sillasf[:, 0],axis=1) , np.expand_dims(sillasf[:, 3] - sillasf[:, 1],axis=1)),axis=1).astype('int32')
              print('Centros Fusión:\n',sfcentros)
              print('Dimensiones Fusión:\n',sfdims)
              #Obtener dimensión mínima de sillas Fusionadas
              #TODO: ALMACENAR JUNTO CON UBICACIONES DE SILLAS
              sfdimmin = np.amin(sfdims, axis=0).reshape(1,2)
              print('Dimensión Mínima Fusión:\n',sfdimmin)

              #Dividir centros de rectángulos con dimensión mínima 
              # para obtener una identificación unívoca
              sfident = np.trunc(np.true_divide(sfcentros,sdimmin))
              print('Identificación de Sillas Fusión (DimMin Original ',sdimmin[0],'):\n Silla > Centro -> Coordenada')
              for p in range(sfident.shape[0]):
                  print(sillasf[p],' > ',sfcentros[p],'->',sfident[p])
              #print('Identificación de Sillas Fusión (DimMin Original):\n',sfident)
              #Remover identificaciones duplicadas
              sfident,sfunqIDX,sfdupIDX,sfdupCOUNT = np.unique(sfident, return_index=True, return_inverse=True, return_counts=True, axis=0)
              print('Identificación de Sillas Fusión (no duplicados):\n',sfident)
              #print('unqIDX',sfunqIDX,'\ndupIDX',sfdupIDX,'\ndupCOUNT',sfdupCOUNT)
              print('Nota: la Fusión previa con Dimensión Mínima original evita los registros duplicados')

              #Dividir centros de rectángulos con dimensión mínima 
              # para obtener una identificación unívoca
              sfident = np.trunc(np.true_divide(sfcentros,sfdimmin))
              print('Identificación de Sillas Fusión (DimMin Fusión ',sfdimmin[0],'):\n Silla > Centro -> Coordenada')
              for p in range(sfident.shape[0]):
                  print(sillasf[p],' > ',sfcentros[p],'->',sfident[p])
              #print('Identificación de Sillas Fusión (DimMin Fusion):\n',sfident)
              #Remover identificaciones duplicadas
              sfident,sfunqIDX,sfdupIDX,sfdupCOUNT = np.unique(sfident, return_index=True, return_inverse=True, return_counts=True, axis=0)
              print('Identificación de Sillas Fusión (no duplicados):\n',sfident)
              #print('unqIDX',sfunqIDX,'\ndupIDX',sfdupIDX,'\ndupCOUNT',sfdupCOUNT)

#               # EL SIGUIENTE PASO ES INNECESARIO, SOLO ÚTIL PARA OBTENER
#               # RECTÁNGULO DE UBICACIONES IDENTIFICATORIAS
#               #Expandir matriz y vector mínimo para calcular rectángulos Identificados
#               m1 = np.concatenate((sident,sident+1),axis=1)
#               m2 = np.concatenate((sdimmin,sdimmin),axis=1)
#               sidrect = m1 * m2
#               print('Rectángulos ID de Sillas:\n',sidrect)
#             # Visualization of the results of a detection.
#             vis_util.visualize_boxes_and_labels_on_image_array(
#             image_np,
#             np.squeeze(boxes),
#             np.squeeze(classes).astype(np.int32),
#             np.squeeze(scores),
#             category_index,
#             use_normalized_coordinates=True,
#             line_thickness=8)

#             cv2.imshow('object detection', image_np)
#         else:
#           print('Frame no recibido.')
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#           break
# except IOError as e:
#     print(time.now(), "Error abriendo socket: ", ipcamUrl)
# except cv2.error as e:
#     print(time.now(), "Error CV2: ", e)
# except KeyboardInterrupt as e:
#     print(time.now(), "Detenido por teclado.")
# #except BaseException as e:
# #    print(time.now(), "Error desconocido: ", e)
# ##    if e.number == -138:
# ##        print("Compruebe la conexión con '" + ipcamUrl + "'")
# ##    else:
# ##        print("Error: " + e.message)
# finally:
#     cap.release()
#     cv2.destroyAllWindows()

