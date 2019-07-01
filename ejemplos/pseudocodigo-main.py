
from datetime import datetime as time
import numpy as np
import cv2

def main():
    def init():
        Bancas.init()
        #obtiene identificación de bancas, ubicación y cuadro mínimo de cámara
        Bancas.readConfig()
        #configura conexión a cámaras
        Camara.readConfig()
        ###Si nos recuperamos de falla no debería pisar lo preexistente
        ##BD.setBancas()
        ##BD.setReconociendo(true)
        
    def keyStop():
        #uso variable "estática" para nuevos llamados a la función
        if not hasattr(func,"exit") or not func.exit:
            func.exit = false
            if cv2.waitKey(25) & 0xFF == ord('q'):
                BD.setReconociendo(false)
                func.exit = true
        return func.exit
    
    #bucle infinito (funciona en background como servicio)
    while not keyStop():
        #espera orden de iniciar a reconocer
        while (not BD.getReconociendo()):
            #Optimización: reducir consultas durante la noche
            CNF_OPER_MIN_TIME = 8; CNF_OPER_MAX_TIME = 23; 
            TIME_JUMP_DAY = 20; TIME_JUMP_NIGHT = 1800;
            tcheck = time.now()
            if (tcheck >= time(CNF_OPER_MAX_TIME,00) 
                or tcheck <= time(CNF_OPER_MIN_TIME,00)):
                time.sleep(TIME_JUMP_NIGHT)
            else:
                time.sleep(TIME_JUMP_DAY)
        
        frames = []
        #Optimización: Reconocer menos frames de los recibidos
        FRAMES_OMITIDOS = 10; frameI = 0; 
        #Optimización: Reducir consultas a BBDD
        VUELTAS_SIN_BD = FRAMES_OMITIDOS * 10; vueltaN = 0; 
        while ((vueltaN < VUELTAS_SIN_BD or BD.getReconociendo()) 
            and not keyStop()):
            vueltaN += 1
            #comprueba conexión y captura errores de cámara
            frames = Camara.readFrame()
            tstamp = time.now()
            if (frameI < FRAMES_OMITIDOS):
                frameI += 1 #lee pero no procesa
            else:
                frameI = 0
                #lee imagen, procesa RN y obtiene cuadros
                frm = RN.recognizeFrames(frames)
                #agrupa cuadros coincidentes/solapados
                RN.groupSimilar() 
                #define identificación de cuadros reconocidos
                newState = Bancas.identify(RN.getFrames())
                #agrega estado de bancas, descartando cuadros inútiles
                Bancas.addState(newState, tstamp) 
                #compara ultimos estados y define ocupacion
                Bancas.defineOccupied() 
                #obtener cambios de ocupación de bancas
                news = Bancas.getLastChanges()
                #notificar estados cambiados 
                BD.updateBancas(news)
                ## Acá podríamos notificar directamente al sistema de HCDP

            ## Acá se puede mostrar frames y hacer stream del mismo
