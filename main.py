#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

import bancas as bca
import camaras as cam



class CamServer():
    from datetime import datetime as time
    def __init__(self, nombre="", dbConfig={}):
        self.FRAMES_OMITIDOS = 10 #Análisis en LAN: frames{fluido,delay}= 4{si,>4"} 7{si,<1"} 10{si,~0"}
        self.nom = nombre
        self.source = cam.DBSource(self,dbConfig)
        #Procesa setup y estado suspend    
        self.processNewState(Status.SUSPENDING)
        
    def setup(self):
        #obtiene configuración de servidor, salvo que no exista y toma la BASE
        newStatus, self.conf = self.source.readSvrInfo()
##TODO: chequear newStatus no es asignado
##TODO: chequear configuración cargada correctamente
        #obtiene configuración de cámaras (ip/url,bancas)
        c = self.source.readCamInfo() ##debería indicar cams a buscar
        #obtiene estado de bancas (útil al recuperar post falla)
        b = self.source.readOcupyState()
        
        self.cams = cam.Camaras(c)
        #comprobar conexión de cámaras
        self.cams.checkConn()
        self.bancas = bca.Bancas(b,c)
        #iniciar red neuronal
        PATH_TO_CKPT = 'modelo_congelado/frozen_inference_graph.pb'
        PATH_TO_LABELS = os.path.join('configuracion', 'label_map.pbtxt')
        PATH_TO_TEST_IMAGES_DIR = 'img_pruebas'
        TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
        self.rn = RN(PATH_TO_CKPT,PATH_TO_LABELS,TEST_IMAGE_PATHS)

        
        
    def start(self):
        self.status = Status.WORKING
    
    def suspend(self):
        self.status = Status.SUSPENDED
    
    ''' Procesar nuevo estado de servidor (control externo)
    - Se recibe status=[STARTING,RESTARTING,SUSPENDING]
    - Se procesan funciones: setup, start, suspend
    - Solo procesa cuando el newStatus difiere del actual '''
    def processNewState(self, newStatus=Status.OFF):
        if (self.status != newStatus):
            if (self.status == Status.OFF 
                or newStatus == Status.RESTARTING):
                #Recargar configuracion servidor
                self.setup() 
            if (self.status in [Status.OFF,Status.SUSPENDED]
                and newStatus in [Status.STARTING,Status.RESTARTING]):
                #iniciar servidor / comenzar reconocimiento
                self.start()
            if (self.status == Status.WORKING
                and newStatus == Status.SUSPENDING):
                #suspender servidor / detener reconocimiento
                self.suspend()

    def keyStop():
        #uso variable "estática" para nuevos llamados a la función
        if not hasattr(keyStop,"exit") or not keyStop.exit:
            keyStop.exit = false
            if cv2.waitKey(25) & 0xFF == ord('q'):
                BD.setReconociendo(false)
                keyStop.exit = true
        return keyStop.exit

    ''' Proceso background de servidor '''
    def runService(self):
        try:
            i = self.FRAMES_OMITIDOS
            #bucle infinito (funciona en background como servicio)
            while not keyStop():
                self.cams.captureFrame()
                if (i < FRAMES_OMITIDOS):
                    i += 1
                else:
                    i = 0 
                    
                    rect = self.rn.detect(self.cams.frames, "person", 
                                          float(self.conf["ppersona"]))
                    self.bancas.addDetection(rect)
                    newstate = self.bancas.evaluateOcupy()
                    
            
        except IOError as e:
            print(time.now(), "Error abriendo socket: ", ipcamUrl)
        except cv2.error as e:
            print(time.now(), "Error CV2: ", e)
        #    if e.number == -138:
        #        print("Compruebe la conexión con '" + ipcamUrl + "'")
        #    else:
        #        print("Error: " + e.message)
        except KeyboardInterrupt as e:
            print(time.now(), "Detenido por teclado.")
            
        except BaseException as e:
            print(time.now(), "Error desconocido: ", e)

if __name__ == "__main__":
    serverName = "SVR1"
    dbConfig = {'user':"usher",
                'passwd':"usher",
                'svr':"localhost",
                'db':"usher_rec"}
    sys.path.append("..")
    rnConfig = ['modelo_congelado/frozen_inference_graph.pb',
                os.path.join('configuracion', 'label_map.pbtxt'),
                ]
    NUM_CLASSES = 90
    FRAMES_OMITIDOS = 10 #Análisis en LAN: frames{fluido,delay}= 4{si,>4"} 7{si,<1"} 10{si,~0"}

  PATH_TO_TEST_IMAGES_DIR = 'img_pruebas'
  TEST_IMAGE_PATHS = [ os.path.join('img_pruebas', 'image{}.jpg'.format(i)) for i in range(1, 3) ]

    svr = CamServer(serverName, dbConfig) #(sourceDB|sourceFile)
    svr.runService()
else:
    print("Ejecutando desde ", __name__)