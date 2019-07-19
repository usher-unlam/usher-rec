#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
# Importación del módulo de detección de objetos.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class RN():
    def __init__(self, ckptPB="", labelsPBTXT="", testIMGS=[]):
        self.PATH_TO_CKPT = ckptPB
        self.PATH_TO_LABELS = labelsPBTXT
        self.TEST_IMAGE_PATHS = testIMGS
        self.IMAGE_SIZE = (12, 8)
        self.NUM_CLASSES = 90
        #variables detección por cámara
        self.boxes = []; self.scores = []; self.classes = []; self.num = []
        
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as self.sess:
                self.sess = tf.Session()
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                print("Detection Classes:",self.detection_classes)
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    ##TODO: comprobar errores en carga de RN
    
    def detect(self, frames, classFilter="person", scoreFilter=0.5):
        det = 0
        rect = []
    #        rcent = []
    #        rdims = []
        self.boxes = []; self.scores = []; self.classes = []; self.num = []
        for k,f in frames.items():
            if f:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(f, axis=0)
                # Actual detection.      
                ##(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
                # Actual detection.      
                self.boxes[k], self.scores[k], self.classes[k], self.num[k] = self.sess.run(
                        [self.detection_boxes, self.detection_scores, 
                         self.detection_classes, self.num_detections],
                         feed_dict={self.image_tensor: image_np_expanded})
                det += 1
                
                box = np.squeeze(self.boxes[k])
                #Alto del frame en pixeles
                height = np.size(f, 0)
                #Ancho del frame en pixeles
                width = np.size(f, 1)
                
                #cargar rectangulos segun filtros class y score                
                for index, value in enumerate(self.classes[k][0]):
                    if self.scores[k][0,index] > scoreFilter:
                        if self.category_index.get(value).get('name') == classFilter:
                            print("Class Value for ",classFilter,":", value, box[index])
                            ymin = (int(box[index,0] * height))
                            xmin = (int(box[index,1] * width))
                            ymax = (int(box[index,2] * height))
                            xmax = (int(box[index,3] * width))
                            #operado como (Y,X) a diferencia de Rectangle
                            rect[k].append( (box[index] * [height,width,height,width]).astype(int))
    ##TODO: calcular centro y dimensiones de forma matricial
                            #Prueba con (Ycentro,Xcentro) y (Alto,Ancho)
    #                            cbox = (int((ymin+ymax)/2),int((xmin+xmax)/2)) #centro:(Y,X)
    #                            rcent[k].append(cbox)
    #                            dbox = (ymax-ymin,xmax-xmin)
    #                            rdims[k].append(dbox) #dimension: (alto,ancho)
                            #print('Silla ',index,' [Xmin,Ymin,Xmax,Ymax]=',rbox,' Centro=',cbox,' Dimensiones:',dbox)
        return rect
    
    #Intersection Over Union (intersection / union) available in mrcnn.utils.compute_overlaps()
    @staticmethod
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
    
    @staticmethod
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
            overlaps[:, i] = RN.compute_iou(box2, boxes1, area2[i], area1)
        return overlaps
    
    @staticmethod
    def fusion(a, b):  
        # espera (ymin,xmin,ymax,xmax) para a y b
        return (min(a[0], b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3]))


class Ubicacion():
    def __init__(self, bstat, cams):
        self.estado = bstat
        self.estados = [] #histórico de estados de ocupación
        #Obtener referencias de ubicaciones x cámara, Ymin,Xmin,Ymax,Xmax x ubicacion
        self.ubicacionesCam, self.yxyxCam = cams.getUbicacionesFromCams()

    ''' Agregar reconocimiento y evaluar estado contra ubicaciones'''
    def addDetection(self, rect):
        for k,r in rect.items():
            #recorro camaras y sus rectángulos reconocidos
            if len(r) > 0:
                #obtiene recuadro mínimo desde configuración de ubicacion
                dimmin = np.array(self.ubicacionesCam[k][0]["minUbicacion"])
                print('Dimensión Mínima:',dimmin.shape,'\n',dimmin)
                #calcular centros de rectángulos
                pass
                ##obtengo centros
                ##obtengo ubicaciones por cámara
                ##identificar rectángulos con minUbicacion por cámara
                ##calcular iou y overlap con ubicaciones de la cámara
    
    ''' Identifica un rectángulo en una cámara específica
        out: coordenadas=(Y * H_minUbicacion, X * W_minUbicacion)'''
    def identify(self, cam, rect):
        dimmin = np.array(self.ubicacionesCam[cam][0]["minUbicacion"])
        cent = np.trunc(np.concatenate(
                (np.expand_dims(
                        np.divide(rect[:, 0] + rect[:, 2],2),
                        axis=1) , 
                np.expand_dims(
                        np.divide(rect[:, 1] + rect[:, 3],2),
                        axis=1)
                ),axis=1))
        ident = np.trunc(np.true_divide(cent,dimmin))
        return ident
           
    def evaluateOcupy(self):
        pass
