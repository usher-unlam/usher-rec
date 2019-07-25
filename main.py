#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv2.cv2 as cv2

import ubicacion as ubi
import conector as cn

from datetime import datetime as time
from datetime import timedelta as delta

class CamServer():
    def __init__(self, nombre="", dbConfig={}):
        #
        self.conf = {"ubicaciones": 92, "frecCNN": 20, "fpsCam": 40, "fpsCNN": 4, 
                    "pbanca": 0.3, "ppersona": 0.5, "pinterseccion": 0.7, "psolapamiento": 0.5, 
                    "CONN_TIMEOUT": 0.6, "CONN_CHECK_TIMEOUT": 5 , 
                    "DB_TIMEOUT" : { "CONNECT": 3, "STATUS_READ": 4000, "STATUS_WRITE": 1000 }} 
        
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
        self.rn = ubi.RN(PATH_TO_CKPT,PATH_TO_LABELS,TEST_IMAGE_PATHS)

        # Establecer conexion con BBDD
        self.source = cn.DBSource(dbConfig,self.conf["DB_TIMEOUT"],self)
        # Comprobar conexión con BBDD
        if not self.source.connect():
            print("Error de conexion a BBDD. Compruebe los datos de conexion.")
            exit(1)
        else:
        ##TODO: chequear conexion correcta con BBDD
            # Procesa setup obteniendo ultimo estado (recuperación post falla)
            currStatus = self.setup()
            # Define estado a procesar segun estado previo (Status.SUSPENDING por defecto) 
            newStatus = cn.Status.SUSPENDING if currStatus in [cn.Status.OFF,cn.Status.RESTARTING] else cn.Status(2 * (int(currStatus) // 2))
            print("BD", currStatus, "=> procesar", newStatus)
            # Definir nuevo estado y guardar en BBDD
            self.processNewState(newStatus) 
        
    def setup(self):
        print("Configurando servidor",self.nombre)
        # Obtiene configuración de servidor, salvo que no exista y toma la BASE
        newStatus, newConf = self.source.readSvrInfo()
        if newConf == {}:
            # Falló recuperando información de servidor
            print("> se mantiene misma configuracion")
            newStatus = self.status
            newConf = self.conf
        
        self.conf = newConf
        # Actualiza configuracion BBDD
        self.source.setup(self.conf["DB_TIMEOUT"])

    ##TODO: chequear newStatus no es asignado
    ##TODO: chequear configuración cargada correctamente
        #obtiene configuración de cámaras (ip/url,ubicaciones)
        c = self.source.readCamInfo() ##debería indicar cams a buscar
        #obtiene estado de ubicaciones (útil al recuperar post falla)
        b = self.source.readOcupyState()
        
        self.cams = cn.Camaras(c,self.conf["CONN_TIMEOUT"],self.conf["CONN_CHECK_TIMEOUT"])
        #comprobar conexión de cámaras por primera vez
        self.cams.checkConn()
        self.ubicaciones = ubi.Ubicacion(b,self.cams)

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
                    if (i < self.conf["frecCNN"] or i < toutDiff):
                        if i == 0:
                            # Garantizar FPS de captura y FREC frames sin procesar
                            tout = time.now() - self.cams.getLastCapture()
                            toutDiff = (tout.total_seconds() * self.conf["fpsCam"]) - 1
                            #print(tout,time.now(),toutDiff)
                            #tout1 = time.now() - delta(milliseconds=1000/self.conf["fpsCam"])
                            #tout2 = time.now() - delta(milliseconds=1000/self.conf["fpsCNN"])
                        i += 1
                        self.cams.escapeFrame()
                    else:
                        self.cams.captureFrame()
                        print("Frames capturados:",len(self.cams.frames),"de",len(self.cams.cams), " camaras (",i,"descartados)")
                        i = 0 
                        if(len(self.cams.frames)):
                            frame = list(self.cams.frames.values())[0]
                            print("-> Procesando frame >",list(self.cams.frames)[0])
  ####                  rect = self.rn.detect(self.cams.frames, "personaSentada", 
  ####                                        float(self.conf["ppersona"]))
  ####                  self.ubicaciones.addDetection(rect)
  ####                  # cada N detecciones o X tiempo
  ####                      newstate = self.ubicaciones.evaluateOcupy()
                # Si WORKING, solo comprueba estado al capturar, sino, siempre
                if (self.status is not cn.Status.WORKING
                    or i == 0):
                    # Obtener, procesar y actualizar estado en BBDD
                    res, newStatus = self.source.readSvrStatus(self.status)
                    if res:
                        self.processNewState(newStatus)
        except IOError as e:
            print("Error IOError que no capturado correctamente.")
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
    serverName = "SVR1"
    dbConfig = {'user':"usher",
                'passwd':"usher101",
                'svr':"usher.sytes.net",
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
