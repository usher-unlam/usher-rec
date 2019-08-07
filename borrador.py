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

    

try:


      while True:
  

            
            Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

            wmin,wmax = (0,0)
            if len(sillas) > 0:
              print('Sillas:',sillas)
              print('Centros:',scentros)
              print('Dimensiones:',sdims)
              
              #Obtener dimensión mínima de sillas 
              #TODO: ALMACENAR JUNTO CON UBICACIONES DE SILLAS
              sdimmin = np.amin(sdims, axis=0).reshape(1,2)
              print('Dimensión Mínima:',sdimmin)
              ##sdminrect = np.concatenate(([[0,0]],sdimmin),axis=1)
              ##print(sdminrect)
              
              #TODO: fusionar rectángulos solapados de sillas en uno mayor
              sillas = np.array(sillas)
              soverlap = compute_overlaps(sillas,sillas)
              print('Solapamiento Sillas:\n', soverlap)
              
              #Dividir centros de rectángulos con dimensión mínima 
              # para obtener una identificación unívoca
              sident = np.trunc(np.true_divide(scentros,sdimmin))
              print('Identificación de Sillas:\n',sident)
              #Remover identificaciones duplicadas
              sident = np.unique(sident, axis=0)
              print('Identificación de Sillas (no duplicados):\n',sident)
              
              # EL SIGUIENTE PASO ES INNECESARIO, SOLO ÚTIL PARA OBTENER
              # RECTÁNGULO DE UBICACIONES IDENTIFICATORIAS
              #Expandir matriz y vector mínimo para calcular rectángulos Identificados
              m1 = np.concatenate((sident,sident+1),axis=1)
              m2 = np.concatenate((sdimmin,sdimmin),axis=1)
              sidrect = m1 * m2
              print('Rectángulos ID de Sillas:\n',sidrect)
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
        else:
          print('Frame no recibido.')
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
except IOError as e:
    print(time.now(), "Error abriendo socket: ", ipcamUrl)
except cv2.error as e:
    print(time.now(), "Error CV2: ", e)
except KeyboardInterrupt as e:
    print(time.now(), "Detenido por teclado.")
#except BaseException as e:
#    print(time.now(), "Error desconocido: ", e)
##    if e.number == -138:
##        print("Compruebe la conexión con '" + ipcamUrl + "'")
##    else:
##        print("Error: " + e.message)
finally:
    cap.release()
    cv2.destroyAllWindows()

