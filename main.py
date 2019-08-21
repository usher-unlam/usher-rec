#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv2.cv2 as cv2

import ubicacion as ubi
import conector as cn
from stream import CamStream

import time as t
from datetime import datetime as time, timedelta as delta

from textwrap import wrap

class CamServer():
    def __init__(self, nombre="", dbConfig={}):
        self.MAX_ESCAPE_FRAMES = 600
        self.DEF_IGNORE_CHAR = '_'
        self.conf = {"ubicaciones": 92, "frecCNN": 20, "fpsCam": 40, "fpsCNN": 4, 
                    "pbanca": 0.3, "ppersona": 0.5, "pinterseccion": 0.7, "psolapamiento": 0.5, 
                    "CONN_TIMEOUT": 0.6, "CONN_CHECK_TIMEOUT": 5 , 
                    "DB_TIMEOUT" : { "CONNECT": 3, "STATUS_READ": 4000, "STATUS_WRITE": 1000 },
                    "EVAL_LAST_MILLIS": 1500, "CAMERAS": []
                    } 
        
            # frecCNN   Cantidad de frames capturados sin procesar por CNN (para evitar lag)
            # fpsCam    FPS capturados de cámaras
            # fpsCNN    FPS procesados por CNN, podría ser menor a 0 (actualmente sin uso)
        self.nombre = nombre
        self.status = cn.Status.OFF
        print("Iniciando servidor",self.nombre)
        # Iniciar red neuronal
        PATH_TO_CKPT = os.path.join('modelo_congelado','frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join('configuracion', 'label_map.pbtxt')
        #PATH_TO_TEST_IMAGES_DIR = 'img_pruebas'
        #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
        self.rn = None
        self.rn = ubi.RN(PATH_TO_CKPT,PATH_TO_LABELS,TEST_IMAGE_PATHS)
        # Inicializado servidor de stream (webserver)
        self.stream = None
        
        # Establecer conexion con BBDD
        self.source = cn.DBSource(dbConfig,self.conf["DB_TIMEOUT"],self)
        # Comprobar conexión con BBDD
        if not self.source.connect():
            print("Error de conexion a BBDD. Compruebe los datos de conexion.")
            exit(1)
        else:
            # Procesa setup obteniendo ultimo estado (recuperación post falla)
            currStatus = self.setup()
            # Define estado a procesar segun estado previo (Status.SUSPENDING por defecto) 
            newStatus = cn.Status.SUSPENDING if currStatus in [cn.Status.OFF,cn.Status.RESTARTING] else cn.Status(2 * (int(currStatus) // 2))
            print("BD", currStatus, "=> procesar", newStatus)
            # Definir nuevo estado y guardar en BBDD
            self.processNewState(newStatus) 

    def getStatus(self):
        stat = self.status
        now = time.now()
        stat = { "name": self.nombre, 
                "update": now.strftime("%Y-%m-%d %H:%M:%S"),
                "status": int(stat),
                "statdesc": stat.name }
        return stat
        
    def setup(self):
        print("Configurando servidor",self.nombre)
        # Obtiene configuración de servidor, salvo que no exista y toma la BASE
        newStatus, newConf = self.source.readSvrInfo()
        if newConf == {}:
            # Falló recuperando información de servidor
            print("> se mantiene misma configuracion")
            newStatus = self.status
            newConf = self.conf
        
        # Actualizar diccionario de configuracion (se reemplazan valores coincidentes)
        self.conf.update(newConf)
        # Obtener ID de Clase a detectar 
        self.className = "person"
        self.classId = self.rn.getClassId(self.className)
        if self.classId is None:
            pass
            print("Error hallando id de clase o categoría a detectar:",self.className)
        # Actualiza configuracion BBDD
        self.source.setup(self.conf["DB_TIMEOUT"])

    ##TODO: chequear newStatus no es asignado
    ##TODO: chequear configuración cargada correctamente
        #obtiene configuración de cámaras (ip/url,ubicaciones)
        c = self.source.readCamInfo(self.conf["CAMERAS"]) ##debería indicar cams a buscar
        #obtiene estado de ubicaciones (útil al recuperar post falla)
        b = self.source.readOcupyState()
        # Estado por defecto cuando no hay registro previo en BBDD
        if len(b) == 0:
            b = wrap(self.DEF_IGNORE_CHAR *int(self.conf["ubicaciones"]) ,1)
        
        self.cams = cn.Camaras(c,self.conf["CONN_TIMEOUT"],self.conf["CONN_CHECK_TIMEOUT"])
        #comprobar conexión de cámaras por primera vez
        self.cams.checkConn()
        self.ubicaciones = ubi.Ubicacion(b,self.cams,self.conf["EVAL_LAST_MILLIS"],self.DEF_IGNORE_CHAR)

        #iniciar servidor de stream (webserver)
        if self.stream is None:
            self.stream = CamStream()
        # Configurar servidor de stream e iniciar (no afecta si ya se esta ejecutando)
        self.stream.setup(self, self.cams)
        # Inicia servidor de stream 
        # self.stream.startStream() #no es necesario
        # t.sleep(15)
        # # Detener thread de stream (Prueba)
        # self.stream.stopStream()
        # # self.stream.startStream() #no es necesario
        # t.sleep(15)
        # # Detener thread de stream (Prueba)
        # self.stream.startStream()

        return newStatus
        
    def start(self):
        self.status = cn.Status.WORKING
    
    def suspend(self):
        self.status = cn.Status.SUSPENDED
    
    ''' Procesar nuevo estado de servidor (control externo)
    - Se recibe status=[STARTING,RESTARTING,SUSPENDING]
    - Se procesan funciones: setup, start, suspend, ...
    - Solo procesa cuando el newStatus difiere del actual
    - Actualiza estado en BBDD '''
    def processNewState(self, newStatus=cn.Status.OFF):
        forceWrite = False
        if (self.status != newStatus):
            forceWrite = True
            print("Nuevo Estado: actual (",int(self.status),",",str(self.status),") >> nuevo (",int(newStatus),",",str(newStatus),")")
            if (self.status != cn.Status.OFF 
                and newStatus == cn.Status.RESTARTING):
                #Recargar configuracion servidor (si es OFF, setup se omite)
                self.setup() 
            if (self.status in [cn.Status.OFF,cn.Status.SUSPENDED]
                and newStatus in [cn.Status.STARTING]): #,cn.Status.RESTARTING
                #iniciar servidor / comenzar reconocimiento
                self.start()
            if (self.status in [cn.Status.OFF,cn.Status.WORKING]
                and newStatus == cn.Status.SUSPENDING):
                #suspender servidor / detener reconocimiento
                self.suspend()
        self.source.writeSvrStatus(self.nombre, self.status, self.conf, forceWrite)

    def keyStop(self):
        #uso variable "estática" para nuevos llamados a la función
        if not hasattr(CamServer.keyStop,"exit"):
            setattr(CamServer.keyStop,'exit', False)
        if not getattr(CamServer.keyStop,'exit'):
            if cv2.waitKey(25) & 0xFF == ord('q'):
                self.processNewState(cn.Status.SUSPENDING)
                setattr(CamServer.keyStop,'exit', True)
        return getattr(CamServer.keyStop,'exit')

    ''' Proceso background de servidor '''
    def runService(self):
        try:
            i = self.conf["frecCNN"]
            toutDiff = i

            # Bucle infinito (funciona en background como servicio)
            while not self.keyStop():
                if self.status is cn.Status.WORKING:
                    if (i < self.conf["frecCNN"] or i < toutDiff) and i < self.MAX_ESCAPE_FRAMES:
                        if i == 0:
                            # Garantizar FPS de captura y FREC frames sin procesar
                            tout = time.now() - self.cams.getLastCapture()
                            toutDiff = (tout.total_seconds() * self.conf["fpsCam"]) - 1
                            #print(tout,time.now(),toutDiff)
                            #tout1 = time.now() - delta(milliseconds=1000/self.conf["fpsCam"])
                            #tout2 = time.now() - delta(milliseconds=1000/self.conf["fpsCNN"])
                        i += 1
                        self.cams.escapeFrame()
                        # permitir a otro thread trabajar
                        t.sleep(0)
                    else:
                        
                        self.cams.captureFrame()
                        
                        print("")   
                        print("------------ NUEVO CICLO ----------------") 
                        print("Frames capturados:",len(self.cams.frames),"de",len(self.cams.cams), " camaras (",i,"descartados)")
                        i = 0 
                        if len(self.cams.frames) > 0:
                            if self.rn.canDetect():
                                ##frame = list(self.cams.frames.values())[0]
                                #print("-> Procesando frame >",list(self.cams.frames)[0])
                                
                                rect = self.rn.detect(self.cams.frames, 
                                                    classFilterName=self.className, classFilterId=self.classId, 
                                                    scoreFilter=float(self.conf["ppersona"]))
                                     
                                                        
                                self.ubicaciones.addDetection(rect)
                                
                                # Evalúa ocupación cada X tiempo, analizando un grupo de detecciones
                                tnewstate, newstate, isnew = self.ubicaciones.evaluateOcupy()
                                if isnew:
                                    #graba nuevo estado en BBDD
                                    print("")
                                    print("GRABANDO EN BDD...",end="")
                                    self.source.writeOcupyState(tnewstate,newstate)
                                 
                            else:
                                print("Advertencia: RN ocupada (no detectará)")
                # Si WORKING, solo comprueba estado al capturar, sino, siempre
                if (self.status is not cn.Status.WORKING
                    or i == 0):
                    # Obtener, procesar y actualizar estado en BBDD
                    res, newStatus = self.source.readSvrStatus(self.status)
                    if res:
                        self.processNewState(newStatus)
        except IOError as e:
            print("Error IOError no capturado correctamente.")
            #print(time.now(), "Error abriendo socket: ", ipcamUrl)
        except cv2.error as e:
            print(time.now(), "Error CV2: ", e)
        #    if e.number == -138:
        #        print("Compruebe la conexión con '" + ipcamUrl + "'")
        #    else:
        #        print("Error: " + e.message)
        except KeyboardInterrupt as e:
            print(time.now(), "Detenido por teclado.")
            
    #    except BaseException as e:
    #        print(time.now(), "Error desconocido: ", e)

if __name__ == "__main__":
    ##TODO: recibir lo siguiente como parámetros de entrada
    serverName = "SVR1"
    serverName = "TEST"
    dbConfig = {'user':"usher",
                'passwd':"usher101",
                'svr': "usher.sytes.net",
                'db':"usher_rec"}
    sys.path.append("..")
    rnConfig = [
                os.path.join('modelo_congelado', 'frozen_inference_graph.pb'),
                os.path.join('configuracion', 'label_map.pbtxt'),
                ]
    NUM_CLASSES = 90
    FRECUENCIA_CNN = 10 #Análisis en LAN: frames{fluido,delay}= 4{si,>4"} 7{si,<1"} 10{si,~0"}

    PATH_TO_TEST_IMAGES_DIR = 'img_pruebas'
    TEST_IMAGE_PATHS = [ os.path.join('img_pruebas', 'image{}.jpg'.format(i)) for i in range(1, 3) ]

    svr = CamServer(serverName, dbConfig) #(sourceDB|sourceFile)
    svr.runService()
else:
    print("Ejecutando desde ", __name__)
