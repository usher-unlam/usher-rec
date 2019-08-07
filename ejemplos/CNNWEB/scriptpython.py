import wx
import time
import cv2
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import collections
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from datetime import datetime as time
from urllib.parse import urlparse
import glob
import xml.etree.ElementTree as ET
from collections import namedtuple
import socket
# Importación del módulo de detección de objetos.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import mysql.connector
import threading

class MyFrame(wx.Frame):
    def __init__(self, *args, **kwds):

        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize((812, 522))
        
        self.CaptureWidth = 640
        self.CaptureHeight = 480

        self.num=-1
        self.boxes=0
        self.scores=0
        self.classes=0

 
        #Obtengo la posicion, dentro de la toma completa, de cada ubicacion 
        path_locations='configuracion'
        self.images_location=self.xml_to_locations(path_locations)
        #String para guardar el estado de cada banca:
        # 1 = ocupada
        # 0 = libre
        # ? = indeterminado
        self.locations_state=""
                 
        ipcamUrl = 'http://admin:usher@192.168.1.33:8081'
        ipcam = {}
        ipcamDesc = 'Celular'
        ipcam[ipcamDesc] = urlparse(ipcamUrl)
        print(time.now())
        
        # Prueba la conexión al destino ip
        if len(ipcamUrl) > 5:
          err,errMsg = self.urlTest(ipcam[ipcamDesc].hostname,ipcam[ipcamDesc].port)
          if err > 0:
              print(time.now(),"Falló conexión. ",errMsg)
              exit(1)
        
        try:
          self.capture = cv2.VideoCapture(ipcamUrl)
          self.capture.set(3,self.CaptureWidth) #1024 640 1280 800 384
          self.capture.set(4,self.CaptureHeight) #600 480 960 600 288
          
        
          sys.path.append("..")
        
          # Importación del módulo de detección de objetos.
          from object_detection.utils import label_map_util
          from object_detection.utils import visualization_utils as vis_util
        
          PATH_TO_CKPT = 'modelo_congelado/frozen_inference_graph.pb'
        
          PATH_TO_LABELS = os.path.join('configuracion', 'label_map.pbtxt')
        
          NUM_CLASSES = 90
                  
          self.detection_graph = tf.Graph()
          with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
              serialized_graph = fid.read()
              od_graph_def.ParseFromString(serialized_graph)
              tf.import_graph_def(od_graph_def, name='')
        
        
          label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
          categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
          self.category_index = label_map_util.create_category_index(categories)
                          
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
            #self.capture.release()
            cv2.destroyAllWindows()
        
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
              self.sess = tf.Session()
              self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
              self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
              self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
              self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
              self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

              #Creo un timer para
              #activar la CNN y obtener datos del analisis
              self.timer = wx.Timer(self)
              self.Bind(wx.EVT_TIMER, self.OnTimer)

             
              self.Bind(wx.EVT_CLOSE, self.onClose)

              #Estado del programa
              self.STATE_RUNNING = 1
              self.STATE_CLOSING = 2
              self.state = self.STATE_RUNNING
              
              #Cantidad de ciclos del timer que la CNN no trabaja
              #Esto es para evitar lag
              self.FREC=20
              self.FRECUENCIA_CNN=self.FREC
                
              #Conecto con la BDD
              self.con = mysql.connector.connect(user="root",password="12345678",host="localhost",database="bbdd1")
              self.cursor=self.con.cursor()

              #Seteo cada cuanto tiempo se activará el timer
              self.fps=40
              self.timer.Start(1000./self.fps)    # timer interval
        


    def OnTimer(self, event):
        
        ret, self.image_np = self.capture.read()
        
        if ret == True:
          #print("Captura OK")
          pass
        else:
          print("Falló la captura")
          exit(1)       

        #Conecto con la BDD
        self.con = mysql.connector.connect(user="root",password="12345678",host="localhost",database="bbdd1")
        self.cursor=self.con.cursor()
            
        #Consulto valor de Start (ID=1)
        #Si Start=1 sigue analizando la cnn y grabando en la BDD
        #Si Start!=1 pausa la cnn y no graba en la BDD
        self.cursor.execute("SELECT Start FROM cnn WHERE ID = 1")
        result = self.cursor.fetchall()

        #Consulto valor de Start (ID=2)
        #Si Start=1 continuo ejecutando
        #Si Start!=1 termino el programa
        self.cursor.execute("SELECT Start FROM cnn WHERE ID = 2")
        result2 = self.cursor.fetchall()

        if result2[0][0]==0:
          exit(1) 

        #Si Start=1 grabo en la base de datos el estado de la cnn
        if result[0][0]==1:
          if self.FRECUENCIA_CNN==0: 
              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_np_expanded = np.expand_dims(self.image_np, axis=0)

              # Actual detection.      
              (self.boxes, self.scores, self.classes, self.num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],feed_dict={self.image_tensor: image_np_expanded})
              
              box = np.squeeze(self.boxes)
              #Alto del frame en pixeles
              height = np.size(self.image_np, 0)
              #Ancho del frame en pixeles
              width = np.size(self.image_np, 1)
              
              ##Comparo cada rectangulo del xml con cada box de la CNN
              ##Si el porcentaje de coincidencia es mayor a PORC_INTERSECCION guardo "[OK] "
              ##Si no, guardo "[ ] "
              self.locations_state=""
              personas=0
              Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
              PORC_INTERSECCION=0.3
              
              #Recorro las posiciones del xml
              for j in self.images_location:
                ymin = int(j[1])
                xmin = int(j[0])
                ymax = int(j[3])
                xmax = int(j[2])
                area_xml=(ymax-ymin)*(xmax-xmin)
                rxml = Rectangle(xmin, ymin, xmax, ymax)
                #Para cada posicion recorro las boxes buscando coincidencia
                coincide=False
                for index,value in enumerate(self.classes[0]):
                 if self.scores[0,index] > 0.3:
                   if self.category_index.get(value).get('name')=="person":
                     ymin = (int(box[index,0]*height))
                     xmin = (int(box[index,1]*width))
                     ymax = (int(box[index,2]*height))
                     xmax = (int(box[index,3]*width))
                     rbox = Rectangle(xmin, ymin, xmax, ymax)
                     area_interseccion=self.area(rxml, rbox)
                     if(area_interseccion!=None):
                       if area_interseccion>(PORC_INTERSECCION*area_xml):
                         coincide=True     
                      
                if coincide==True:
                  self.locations_state=self.locations_state+"1"
                  personas+=1
                else:
                  self.locations_state=self.locations_state+"0"
     
              print ("Se detectaron "+str(personas)+" personas\n")
              print (self.locations_state)
              print ("\n")
              
              self.cursor.execute("INSERT INTO estado (estadoBancas) VALUES ('"+self.locations_state+"')")
              self.con.commit()

              self.FRECUENCIA_CNN=self.FREC
          else:
              self.FRECUENCIA_CNN-=1
         
        ###############################################
        self.timer.Start(1000./self.fps)
        event.Skip()
         

 # end of class MyFrame


    #Al cerrar la ventana paro el timer y elimino el frame
    def onClose(self, event):
        if not self.state == self.STATE_CLOSING:
            self.con.close()
            self.state = self.STATE_CLOSING
            self.timer.Stop()
            self.Destroy()    
  

    def urlTest(self,host, port):
        
        out = (0,"")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.6)
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
    #(correspondiente a cada banca) dentro de la toma de video completa y el nro de banca
    def xml_to_locations(self,path):
        locations_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (int(member[4][0].text), #xmin
                         int(member[4][1].text), #ymin
                         int(member[4][2].text), #xmax
                         int(member[4][3].text), #ymax
                         int(member[0].text), #nro banca                        
                         )
                locations_list.append(value)
        return locations_list
    
    #Area de interseccion entre 2 rectangulos
    #Para determinar coincidencia entre las posiciones del xml y las boxes de la CNN
    def area(self,a, b):  # returns None if rectangles don't intersect
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx>=0) and (dy>=0):
            return dx*dy
            
class MyApp(wx.App):

    def OnInit(self):
        
        self.frame = MyFrame(None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True

# end of class MyApp

if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()