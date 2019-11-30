#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime as time
from datetime import timedelta as delta
import numpy as np
import cv2
from numpy.lib import recfunctions as rfn
import tensorflow as tf
# Importación del módulo de detección de objetos.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import threading

class RN():
    def __init__(self, ckptPB="", labelsPBTXT=""):
        self.PATH_TO_CKPT = ckptPB
        self.PATH_TO_LABELS = labelsPBTXT
        self.IMAGE_SIZE = (12, 8)
        self.NUM_CLASSES = 90
        #variables detección por cámara
        self.boxes = []; self.scores = []; self.classes = []; self.num = []
        
        self.working = threading.Lock()
        self.init = threading.Thread(target=self.initialize,name="RNThread") #, args=(index,)
        self.init.start()

    # Proceso extenso paralelizado con thread (demora ~33 segundos)
    def initialize(self):
        if not self.working.locked():
            self.working.acquire()
        print('RN init-start ', time.now())
        try:
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
            print('RN sess default create ', time.now())
             # Ejecutar primer detección ¿para la construccion de la RN? Demora mucho la primer detección
            initFrame = { "init": np.zeros((640, 480, 3)) }
            self.working.release()
            self.detect(initFrame)
        except BaseException as e:
            print('RN sess default create ERROR: ', e, time.now())
        print('RN init-end ', time.now())
        ##TODO: comprobar errores en carga de RN
    
    def getClassId(self, className):
        if self.working.locked():
            print("Error RN trabajando")
            return None
        c = className.upper()
        for i,val in enumerate(self.category_index.values()):
            if val['name'].upper() == c:
                return val['id']
        return None

    def canDetect(self):
        return not self.working.locked()

    # Detecta objetos en frames y filtra por clase indicada
    def detect(self, frames, classFilterName="",classFilterId=None, scoreFilter=0.5):
        det = 0
        rect = {}
        # Obtener id de clase o categoria
        if classFilterId is None:
            classFilterId = self.getClassId(classFilterName)
        self.working.acquire()
    #        rcent = []
    #        rdims = []
        self.boxes = {}; self.scores = {}; self.classes = {}; self.num = {}
        for k,f in frames.items():
            if len(f) > 0:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                
                image_np_expanded = np.expand_dims(f, axis=0)
                
                ### Actual detection.      
                ##(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
                # Actual detection      
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
                for index, score in enumerate(self.scores[k][0]):
                    if score > scoreFilter:
                        if self.classes[k][0, index] == classFilterId:
                # for index, value in enumerate(self.classes[k][0]):
                #     if self.scores[k][0,index] > scoreFilter:
                #         if self.category_index.get(value).get('name') == classFilter:
                            #print("Class Value for ",classFilterName,":", classFilterId, box[index])
                            # ymin = (int(box[index,0] * height))
                            # xmin = (int(box[index,1] * width))
                            # ymax = (int(box[index,2] * height))
                            # xmax = (int(box[index,3] * width))
                            # operado como (Y,X) a diferencia de Rectangle
                            if not k in rect:
                                rect[k] = []
                            rect[k].append((box[index] * [height,width,height,width]).astype(int))
                            #print('Silla ',index,' [Xmin,Ymin,Xmax,Ymax]=',rbox,' Centro=',cbox,' Dimensiones:',dbox)
        self.working.release()
        return rect
    
    @staticmethod
    def __compute_inters(box, boxes, box_area):
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
    @staticmethod
    def compute_intersection(boxes1, boxes2):
        # Area of anchors and GT boxes
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        # Compute intersection to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the Intersection value.
        intersection = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(intersection.shape[1]):
            box2 = boxes2[i]
            intersection[:, i] = RN.__compute_inters(box2, boxes1, area1)
        return intersection

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
    
    # Halla rectángulos con ID duplicado y los fusiona
    @staticmethod
    def fusionDuplicatedId(rect, rectId):
        print('-> Filtrando duplicados')
        # obtener duplicados y unicos
        sident,sunqIDX,sdupIDX,sdupCOUNT = np.unique(rectId, 
                                                    return_index=True, 
                                                    return_inverse=True, 
                                                    return_counts=True, 
                                                    axis=0)
        print(sident)
        # fusionar rectángulos id duplicados
        f = np.zeros((sident.shape[0],rect.shape[1]))
        print('-> Fusionando duplicados') #f
        for i,idup in enumerate(sdupIDX):
            #si no está agregado, lo agrego en la posición idup
            if np.sum(f[idup]) == 0:
                f[idup] = rect[sunqIDX[idup]]
            else:
                f[idup] = RN.fusion(f[idup],rect[sunqIDX[idup]])
        print(f)
        return f,sident

    # Fusiona rectángulos solapados con cierto grado de solapamiento e intersección
    @staticmethod
    def fusionOverlapIntersect(rect):
        print('-> Fusiónando por Solapamiento-Intersección')
        r = np.array(rect)
        rOverlap = compute_overlaps(r,r)
        #Calcular intersección entre sillas
        rInters = compute_intersection(r,r)
        #Fusionar/Extender rectángulos solapados
        solapX = r.shape[0]
        f = []
        #print('Inicio de Fusión/Extensión')
        for i in range(solapX - 1):
            if rInters[i][i] == 1:
                ins = rect[i]
                for j in range(i+1, rOverlap.shape[0]):
                    if rInters[j][j] != 0:
                        #Empleando SOLAPAMIENTO
                        if rOverlap[i][j] > 0.5:
                            ins = fusion(ins,rect[j])
                            sinters[j][j] = 0 #anulo el posterior procesamiento de este elemento
                            print(i+1,',',j+1,'  ', ins, ' x solapamiento')
                        #Empleando INTERSECCIÓN
                        elif max(rInters[i][j],rInters[j][i]) > 0.7:
                            ins = fusion(ins,rect[j])
                            sinters[j][j] = 0 #anulo el posterior procesamiento de este elemento
                            print(i+1,',',j+1,'  ', ins, ' x intersección')
                f.append(ins)
        f = np.array(f)
        return f

    ''' Identifica rectángulos en una cámara específica
        out: (coordY, coordX) = (Y / H_minUbicacion, X / W_minUbicacion)'''
    @staticmethod
    def identify(rect, mindim):
        rect = np.array(rect)
        print('-> Identificando rectangulos')
        # obtiene recuadro mínimo desde parametro de entrada
        dimmin = np.array(mindim)
        print('-> Dimensión Mínima:', dimmin, dimmin.shape)
        # calcular centro de rectángulos
        cent = np.trunc(np.concatenate(
                (np.expand_dims(
                        np.divide(rect[:, 0] + rect[:, 2],2),
                        axis=1) , 
                np.expand_dims(
                        np.divide(rect[:, 1] + rect[:, 3],2),
                        axis=1)
                ),axis=1))
        # identificar, calculando coordenadas
        ident = np.trunc(np.true_divide(cent, dimmin))
        return ident

    ''' Obtiene dimensiones de rectángulo minimo dentro de una coleccion de rectángulos '''
    @staticmethod
    def minRect(rect):
        rect=np.array(rect)
        print('-> Obteniendo rectángulo mínimo')
        # obtener tamaño de rectángulos
        dims = np.concatenate((
                np.expand_dims(rect[:, 2] - rect[:, 0],axis=1), 
                np.expand_dims(rect[:, 3] - rect[:, 1],axis=1)
                ),axis=1).astype('int32')
        print("Dimensiones:",dims)
        # obtener dimensión mínima de rectángulo
        mindim = np.amin(dims, axis=0).reshape(1,2)
        print("Dimensión mínima:", mindim)
        return mindim



class Ubicacion():
    def __init__(self, laststat, cams, evalLastMillis=500, ignoreChar='_'):
        self.EVAL_LASTMILLIS = evalLastMillis
        self.DEF_IGNORE_CHAR = ignoreChar
        self.ocupyState = laststat
        self.states = {} #histórico de estados de ocupación
        self.lastRect = {} #rectangulos reconocidos por RN en ultimos frames agregados (addDetection)
        self.cams = cams
        # Obtener referencias de cámaras x ubicacion, matrices [coordY,coordX] / [Ymin,Xmin,Ymax,Xmax] x cámara
        self.ubiCam, self.camMinFrame, self.camNum, self.camWeight, self.camCoord, self.camYXYX = cams.getUbicacionesFromCams()
        # # Convertir a Numpy array
        # self.camCoord = np.array(self.camCoord)
        # self.camYXYX = np.array(self.camYXYX)
        # Calcular cuadro mìnimo si no estuviera en BBDD
        # Calcular coordenadas si no estuvieran en BBDD
        print("camCoord Antes: ",self.camCoord)
        for cam,num in self.camNum.items():
            if len(self.camMinFrame[cam]) == 0:
                self.camMinFrame[cam] = RN.minRect(self.camYXYX[cam]) 
            self.states[cam] = {"upd": [], "stat": []}
            if len(self.camCoord[cam]) == 0 or len(self.camCoord[cam][0])==0:
                self.camCoord[cam] = RN.identify(self.camYXYX[cam], self.camMinFrame[cam])
                
        self.tlastEval = time.now() # (solo evaluacion)
        print("camCoord Despues: ",self.camCoord)
        
    def count(self):
        return len(self.ocupyState)

    def getLastEval(self):
        return self.tlastEval

    def getNumByCam(self, cam=""):
        if cam == "":
            return self.camNum
        if cam not in self.camNum:
            return np.array([])
        return self.camNum[cam]

    def getCoordByCam(self, cam=""):
        if cam == "":
            return self.camCoord
        if cam not in self.camCoord:
            return np.array([])
        return self.camCoord[cam]

    def getYxyxByCam(self, cam=""):
        if cam == "":
            return self.camYXYX
        if cam not in self.camYXYX:
            return np.array([])
        return self.camYXYX[cam]

    ''' Devuelve un dict con la última detección hecha, separado por cámaras y un timepo "update" '''
    def getLastDetectionByCam(self, cam=""):
        if cam == "":
            return self.lastRect
        if cam not in self.lastRect:
            return np.array([])
        return self.lastRect[cam]
        
    def getLastStateByCam(self, cam=""):
        if cam == "":
            return self.states
        if cam not in self.states:
            return np.array([])
        return self.states[cam]
        

    ''' Agregar reconocimiento y evaluar estado contra ubicaciones'''
    def addDetection(self, rect):
        # self.camMinFrame['cam2'] = [[145, 107]]
        # self.camYXYX['cam2'] = [[163, 279, 441, 484], [143, 292, 289, 482], [ 87, 201, 295, 338], [108, 323, 276, 500], [107, 198, 252, 305]]
        # self.camCoord['cam2'] = RN.identify(self.camYXYX['cam2'], self.camMinFrame['cam2']) 
        # #[[2.,3.],[1.,3.],[1.,2.],[1.,3.],[1.,2.]] # [[10., 19.],[ 7., 19.],[ 6., 13.],[ 6., 20.],[ 5., 12.]]
        # rect = {'cam2': [[163, 279, 441, 484], [143, 292, 289, 482], [ 87, 201, 295, 338], [108, 323, 276, 500], [107, 198, 252, 305]]}
        #for u,c in self.ubicacionesCam:
        #    cam = c[0]
        # print("ubiCoord:\n",self.camCoord['cam1'])
        # print("ubiYXYX:\n",self.camYXYX['cam1'])
        # print("rect:\n",rect)
        update = time.now()
        for k,r in rect.items():
            #recorro camaras y sus rectángulos reconocidos
            if len(r) > 0:
                r = np.array(r)
                ubiCoord = np.array(self.camCoord[k])
                ubiYXYX = np.array(self.camYXYX[k])
            ##TODO: ¿calcular centro y dimensiones de forma matricial?
                                    #Prueba con (Ycentro,Xcentro) y (Alto,Ancho)
            #                            cbox = (int((ymin+ymax)/2),int((xmin+xmax)/2)) #centro:(Y,X)
            #                            rcent[k].append(cbox)
            #                            dbox = (ymax-ymin,xmax-xmin)
            #                            rdims[k].append(dbox) #dimension: (alto,ancho)
            # 
                ##evaluar estado para esta cámara
                est = self.__evaluateOcupy(self.camNum[k],ubiYXYX,ubiCoord, self.camMinFrame[k], r)
                print("Ubicaciones",k,"\n",est)
                #almacenar estado
                self.states[k]["upd"].append(update)
                self.states[k]["stat"].append(est)

        #Actualizar último rectangulo agregado
        self.lastRect = rect
        self.lastRect["update"] = update

    def __evaluateOcupy(self, ubiN, ubiR, ubiC, ubiMin, rect1):
        # coord1 = RN.identify(rect1, ubiMin)
        #calcular iou y overlap con YXYX de ubicaciones de la cámara
        # over1 = RN.compute_overlaps(rect1,ubiR)
        # print('Solapamiento YXYX:\n', over1)
        #Calcular intersección con YXYX de ubicaciones de la cámara
        inter1 = RN.compute_intersection(rect1,ubiR)
        #print('Intersección YXYX:\n', inter1)

        # ##fusionar rectangulos con coord duplicadas
        # rect2,coord2 = RN.fusionDuplicatedId(rect1,coord1)
        # #calcular iou y overlap con YXYX de ubicaciones de la cámara
        # over2 = RN.compute_overlaps(rect2,ubiR)
        # print('Solapamiento YXYX:\n', over2)
        # #Calcular intersección con YXYX de ubicaciones de la cámara
        # inter2 = RN.compute_intersection(rect2,ubiR)
        # print('Intersección YXYX:\n', inter2)
        
        # Definir ocupación de ubicación (ocupado/libre)
        est = np.zeros( (ubiR.shape[0],) )
        # Reconocimiento usando INTERSECCION
        for b in range(ubiR.shape[0]):
            j = 0
            while j<len(rect1) and inter1[j][b] < 0.5:
                j += 1
            if j<len(rect1):
                est[b] = 1
            
        # # Ordenar coordenadas
        # coordtype = [('y', int), ('x', int)]
        # ordUbiC = np.sort(np.array(ubiC,dtype=coordtype),axis=0,order=['y','x'])
        # ordC = np.sort(np.array(coord1,dtype=coordtype),axis=0,order=['y','x'])
        # ordUbiC = np.squeeze(rfn.structured_to_unstructured(ordUbiC[['x']]),axis=None)
        # ordC = np.squeeze(rfn.structured_to_unstructured(ordC[['x']]),axis=None)
        # #print(ordUbiC)
        # #ordUbiC = sorted(self.cams["cam2"].items(), key = lambda kv:(kv[1], kv[0]))
        # #print(ordUbiC)
        # j = 0
        # for i,c in enumerate(ordUbiC):
        #     while j<len(ordC) and bool(np.greater(c,ordC[j]).sum()):
        #         j += 1
        #     if j == len(ordC):
        #         break
        #     if np.array_equal(c, ordC[j]):
        #         print(c, "vs", ordC[j])
        #         est[i] = 1
        #         j += 1
        return est

    # Devuelve la fecha/hora, el estado nuevo calculado, un bool indicando si cambió respecto al estado anterior
    def evaluateOcupy(self):
        DEF_EMPTY_VAL = 9
        tCurrEval = time.now()
        tout = tCurrEval - delta(milliseconds=self.EVAL_LASTMILLIS)
        cambio = False

        if tout > self.tlastEval:
            evaluo = False
            ests = np.full( (len(self.camNum), self.count()), DEF_EMPTY_VAL ) #lleno de DEF_EMPTY_VAL
            # Inicializar pesos por cámara y ubicación
            pesos = np.ones( ests.shape ) #lleno de 1
            pesosInit = np.zeros( (ests.shape[1]) )
            # Evaluar estados de una misma camara
            for c,k in enumerate(self.camNum):
                if len(self.states[k]["stat"]) > 0:
                    #TODO: comparar y filtrar solo fecha >= tout
                    est = np.array(self.states[k]["stat"])
                    # Vaciar listas
                    self.states[k]["upd"].clear()
                    self.states[k]["stat"].clear()
                    # Promedio de cada ubicacion redondeando 0.5 hacia arriba
                    promCam = (np.mean(est, axis=0) + 0.00001).round()
                    # Cargar estado/peso por cada ubicacion de camara
                    for ubi,prom,peso in zip(self.camNum[k],promCam,self.camWeight[k]):
                        u = ubi - 1 # Corrección del índice de array vs número de banca
                        ests[c,u] = prom
                        # inicializar columna de pesos en 0 (no puede hacerse antes x riesgo de división por 0)
                        if not pesosInit[u]:
                            pesos[:,u] = 0
                            pesosInit[u] = 1
                        pesos[c,u] = peso
                    evaluo = True
            if evaluo:
                # Evaluar estados de una misma ubicacion (promedio ponderado redondeando 0.5 hacia arriba)
                currEval = (np.average(ests,axis=0,weights=pesos) + 0.00001).round().astype(int)
                self.tlastEval = tCurrEval
                # Convertir valor a texto/string, omitir ubicaciones no reconocidas
                #estChar = np.array2string(currEval.astype(int),separator='')[1:-1].replace(str(DEF_EMPTY_VAL),self.DEF_IGNORE_CHAR)
                estChar = np.char.replace(currEval.astype(int).astype(str),str(DEF_EMPTY_VAL),self.DEF_IGNORE_CHAR)
                ## Establece 'cambio' solo si difiere de estado anterior
                ## En este caso no se actualizaría el tstamp de 'estado' sino solo el 'update' del servidor
                #cambio = not np.array_equal(self.ocupyState, estChar) 
            else:
                # Si no se detecta algo válido, cambiar las bancas previamente ocupadas como libres
                estChar = np.char.replace(self.ocupyState,'1',self.DEF_IGNORE_CHAR)
            cambio = True
            self.ocupyState = estChar.tolist()
        return self.tlastEval, self.ocupyState, cambio
            #r2 = (np.mean(r,axis=0) + 0.00001).round()
            # >>> p
            # array([[ 1, 10,  0],
            #     [ 1,  1, 10],
            #     [10,  1,  1]])
            # >>> r
            # array([[1, 0, 0],
            #     [1, 1, 0],
            #     [1, 1, 1]])
            # >>> np.sum(p,axis=0)
            # array([12, 12, 11])
            # >>> p1=np.sum(p,axis=0)
            # >>> np.divide(p,p1)
            # array([[0.08333333, 0.83333333, 0.        ],
            #     [0.08333333, 0.08333333, 0.90909091],
            #     [0.83333333, 0.08333333, 0.09090909]])
            # >>> np.average(r,axis=0,weights=p)
            # >>> np.average(r,axis=0,weights=unos) = np.mean(r,axis=0)
            # >>> (np.average(r,axis=0,weights=p) + 0.00001).round()
