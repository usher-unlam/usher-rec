#!/usr/bin/env python
# -*- coding: utf-8 -*-
import abc
from abc import ABCMeta
import mysql.connector
import json
from textwrap import wrap
import socket
import os
from urllib.parse import urlparse
from datetime import datetime as time
from datetime import timedelta as delta
import cv2.cv2 as cv2

from enum import IntEnum
class Status(IntEnum):
    OFF = 1
    STARTING = 2
    WORKING = 3
    SUSPENDING = 4
    SUSPENDED = 5
    RESTARTING = 6
        
class CamStatus(IntEnum):
    OK = 0
    ERR_SOCKET = 1
    ERR_ADDRSS = 2
    ERR_CONNTN = 3
    ERR_CV2CAP = 4
    ERR_NOFILE = 10
    ERR_NOACCS = 11

''' 'nombre': 'cam1', 'minUbicacion': [ANCHOpx, ALTOpx], 
        'ip': '192.168.0.10', 'desc': 'camara del techo', 
        'ubicaciones': [
          {'nro': 1, 'coord': [X1, Y1]}, 
          {'nro': 2, 'coord': [X2, Y2]}, 
 '''
class Camaras():
    CONN_TIMEOUT = 0.6
    CONN_CHECK_TIMEOUT = 5 #si en 5 segundos no tuvo conexión, comprueba de nuevo
    
    def __init__(self, c=[], CONN_TIMEOUT=0.6 ,CONN_CHECK_TIMEOUT=5):
        print("Inicializa Camaras")
        if len(c) > 0:
            self.cams = c
        else:
            self.cams = []
        self.camstat = {}
        self.caps = {}
        self.frames = {}
        self.CONN_TIMEOUT = CONN_TIMEOUT
        self.CONN_CHECK_TIMEOUT = CONN_CHECK_TIMEOUT
        self.tlastTime = time.now() # (captura o escape)
        self.tlastCapt = time.now() # (solo captura)
    
    def getLastTime(self):
        return self.tlastTime

    def getLastCapture(self):
        return self.tlastCapt

    @staticmethod
    def urlTest(host, port):
        out = (CamStatus.OK ,"")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout( Camaras.CONN_TIMEOUT )
        except socket.error as e:
            out = (CamStatus.ERR_SOCKET, "Error creating socket: %s" % e)
        # Second try-except block -- connect to given host/port
        else:
            try:
                s.connect((host, port))
            except socket.gaierror as e:
                out = (CamStatus.ERR_ADDRSS, "Address-related error connecting to server: %s" % e)
            except socket.error as e:
                out = (CamStatus.ERR_CONNTN, "Connection error: %s" % e)
            finally:
                s.close()
        return out

    @staticmethod
    def fileTest(file):
        try:
            f = open(file)
            out = (CamStatus.OK ,"")
            f.close()
        except FileNotFoundError as e:
            out = (CamStatus.ERR_NOFILE, "Archivo no existente: %s" % e)
            print('Error archivo no existente:', file)
        except IOError as e:
            out = (CamStatus.ERR_NOACCS, "Archivo sin permiso: %s" % e)
            print('Error archivo sin permiso para leer:', file)
        return out

    def getUbicacionesFromCams(self):
        ubicaciones = {}
        minFr = {}
        camNum = {}
        camWgt = {}
        coord = {}
        yxyx = {}
        for cam in self.cams:
            k = cam["nombre"]
            if "minUbicacion" in cam:
                minFr[k] = cam["minUbicacion"]
            else:
                minFr[k] = []
            camNum[k] = []
            camWgt[k] = []
            coord[k] = []
            yxyx[k] = []
            for ubi in cam["ubicaciones"]:
                if not ubi["nro"] in ubicaciones:
                    ubicaciones[ubi["nro"]] = {}
                ubicaciones[ubi["nro"]][k] = (cam, ubi["coord"],ubi["yxyx"])
                camNum[k].append(ubi["nro"])
                if "peso" in ubi:
                    camWgt[k].append(ubi["peso"])
                elif "peso" in cam:
                    camWgt[k].append(cam["peso"])
                else:
                    camWgt[k].append(1)
                if "coord" in ubi:
                    coord[k].append(ubi["coord"])
                yxyx[k].append(ubi["yxyx"])
        return (ubicaciones, minFr, camNum, camWgt, coord, yxyx)

    def checkConn(self):
        #self.camstat = [(tstamp,estado)]
        #time.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
        tlimit = time.now() - delta(seconds=Camaras.CONN_CHECK_TIMEOUT)
        for cam in self.cams:
            if (not cam["nombre"] in self.camstat 
                or self.camstat[cam["nombre"]][0] < tlimit):
    ##TODO: Chequear si no es url (ej: 0 o "file.mp4")
                url = urlparse(cam["url"])
                if url.netloc != '':
                    out, msj = Camaras.urlTest(url.hostname,url.port)
                else:
                    out, msj = Camaras.fileTest(url.path)
                self.setCamStat(cam["nombre"], out, msj)
    
    def setCamStat(self,cam="",estado=CamStatus.OK, msj=""):
        self.camstat[cam] = (time.now(), estado, msj)
        #self.caps[cam] = None
    ##TODO: guardar estado en base de datos
        # es necesario eliminar caps ante cualquier error/falla
        if (estado != CamStatus.OK):
            self.caps[cam] = None
    ##TODO: Loguear conexión fallida
    
    def escapeFrame(self):
        return self.captureFrame(False)

    def captureFrame(self, saveTime=True):
        #self.caps = []
        self.frames = {}
        self.tlastTime = time.now()
        if saveTime:
            self.tlastCapt = self.tlastTime
        #comprobar cada cierto TIMEOUT la conexión de cámaras 
        self.checkConn()
        if len(self.cams) == 0:
            print("No hay cámaras activas configuradas.")
        for cam in self.cams:
            if self.camstat[cam["nombre"]][1] == CamStatus.OK:
                try:
                    #crea captura si no había sido creado
                    if (not cam["nombre"] in self.caps 
                        or self.caps[cam["nombre"]] is None):
                        capture = cv2.VideoCapture(cam["url"])
                        capture.set(3,640)
                        capture.set(4,480)
                        self.caps[cam["nombre"]] = capture

                    if (self.caps[cam["nombre"]] 
                        and self.caps[cam["nombre"]].isOpened()):
                        ret, image_np = self.caps[cam["nombre"]].read()
                        if ret:
                            self.frames[cam["nombre"]] = image_np
                            #renovar estado OK de cámara
                            self.setCamStat(cam["nombre"], CamStatus.OK, "frame")
                        else:
                            err = "No se recibió frame: " + cam["nombre"]
                            self.setCamStat(cam["nombre"], CamStatus.ERR_CV2CAP, err)
                    else:
                        err = "No se recibe stream de origen: " + cam["nombre"]
                        self.setCamStat(cam["nombre"], CamStatus.ERR_CV2CAP, err)
                except IOError as e:
                    err = "Error abriendo socket: " + cam["nombre"] + " (" + e + ")"
                    self.setCamStat(cam["nombre"], CamStatus.ERR_SOCKET, err)
                    #print(time.now(), "Error abriendo socket: ", ipcamUrl)
                except cv2.error as e:
                    err = "Error CV2: " + cam["nombre"] + " (" + e + ")"
                    self.setCamStat(cam["nombre"], CamStatus.ERR_CV2CAP, err)
                    #print(time.now(), "Error CV2: ", e)
            #else:
                # Tambien ocurre ante error de usuario / contraseña
            #    print(cam["nombre"], " con error: ",self.camstat[cam["nombre"]], " (compruebe usuario/contraseña)")
                #No setear nuevo estado: self.setCamStat(cam["nombre"], CamStatus.ERR_, err)
        return self.frames
    
class DataSource():
    __metaclass__ = ABCMeta
    def __init__(self, camServer):
        print("Inicia DataSource -",type(self).__name__)
        self.camsvr = camServer
        
    @abc.abstractmethod
    def readSvrInfo(self):
        pass
    @abc.abstractmethod
    def keepAlive(self):
        pass
    @abc.abstractmethod
    def readCamInfo(self):
        pass
    @abc.abstractmethod
    def writeCamInfo(self):
        pass
    @abc.abstractmethod
    def readOcupyState(self):
        pass
    @abc.abstractmethod
    def writeOcupyState(self):
        pass
    @abc.abstractmethod
    def close(self):
        pass
        
class FileSource(DataSource):
    def __init__(self, camServer):
        DataSource.__init__(self, camServer)
        #print("inicia FileSource")
        pass

class DBSource(DataSource):
    #{user="root",passwd="12345678",svr="localhost",db="usher_rec"}
    def __init__(self, connData, timeouts, camServer):
        DataSource.__init__(self, camServer)
        #print("inicia DBSource")
        self.tout = timeouts
        self.connData = connData
        self.conn = None
        self.cursor = None
        ## Realizar primer conexion a BBDD
        #self.connect()

    def connect(self):
        if self.conn is None or not self.conn.is_connected():
            self.cursor = None
            try:
                self.conn = mysql.connector.connect(user=self.connData['user'], 
                                                    password=self.connData['passwd'],
                                                    host=self.connData['svr'],
                                                    database=self.connData['db'],
                                                    connection_timeout= self.tout['CONNECT'])
        ##TODO: capturar errores SQL
                if self.conn and self.conn.is_connected():
                    self.conn.config(connection_timeout=30)
                    self.cursor = self.conn.cursor()
                    if self.cursor:
                        return True
            except mysql.connector.Error as error:
                print("Error de BBDD: {}".format(error), "(", self.connData['svr'], ")")
            finally:
                pass
            return False
        else:
            return True
    
    def setup(self, timeouts):
        self.tout = timeouts

    def readSvrStatus(self, defStat=Status.OFF, forced=False):
        newVal = False
        status = defStat
        tlimit = time.now() - delta(milliseconds=self.tout['STATUS_READ'])
        if not hasattr(DBSource.readSvrStatus, 'update'):
            setattr(DBSource.readSvrStatus, 'update', tlimit)
        if (forced 
            or getattr(DBSource.readSvrStatus, 'update') < tlimit):
            if self.connect():
                try:
                    self.cursor.execute("""SELECT status+0 FROM camserver 
                                        WHERE id = %s""", 
                                        (self.camsvr.nombre,))
                    reg = self.cursor.fetchone()
                    if not reg is None and reg[0] > 0:
                        newVal = True
                        status = Status(reg[0])
                    print("Lee BBDD status",getattr(DBSource.readSvrStatus, 'update'),status)
                except mysql.connector.Error as error:
                    print("Lee BBDD status","Error de BBDD: {}".format(error), "(", self.connData['svr'], ")")
                finally:
                    setattr(DBSource.readSvrStatus, 'update', time.now())
                #self.close()
            else:
                print("Lee BBDD status","ERROR CONEXION A BBDD")
    ##TODO: capturar errores SQL
        return newVal, status
        
    # Actualizar estado y fecha de vivo (keep alive) del servidor
    def writeSvrStatus(self, svrNombre, svrStatus, svrConf, forced=False):
        out = False
        tlimit = time.now() - delta(milliseconds=self.tout['STATUS_WRITE'])
        if not hasattr(DBSource.writeSvrStatus, 'update'):
            setattr(DBSource.writeSvrStatus, 'update', tlimit)
        if (forced 
            or getattr(DBSource.writeSvrStatus, 'update') < tlimit):
            if self.connect():
                try:
                    #(select z.config from camserver as z where z.id='BASE' LIMIT 1)
                    self.cursor.execute("""INSERT INTO camserver (id,alive,status,config) 
                                        VALUES (%s,NULL,%s,%s) 
                                        ON DUPLICATE KEY UPDATE alive=null, status=if(%s,VALUES(status),status)""",
                                        (svrNombre, int(svrStatus), json.dumps(svrConf),forced))
                    out = self.conn.commit()
                    print("Graba BBDD status",getattr(DBSource.writeSvrStatus, 'update'),svrStatus)
                    out = True
                except mysql.connector.Error as error:
                    print("Error de BBDD: {}".format(error), "(", self.connData['svr'], ")")
                except mysql.connector.InterfaceError as error:
                    print("Error de BBDD: {}".format(error), "(", self.connData['svr'], ")")
                finally:
                    setattr(DBSource.writeSvrStatus, 'update', time.now())
                #self.close()
            else:
                print("Graba BBDD status","ERROR CONEXION A BBDD")
    ##TODO: capturar errores SQL
        return out

    def readSvrInfo(self):
        status = Status.OFF
        server = {}
        if self.connect():
            try:
                self.cursor.execute("""SELECT status+0 as status, config FROM camserver 
                                WHERE id in (%s, 'BASE') 
                                ORDER BY alive DESC LIMIT 1""", 
                                (self.camsvr.nombre,))
                reg = self.cursor.fetchone()
                if not reg is None:
                    if reg[0] > 0:
                        status = Status(reg[0])
                    server = json.loads(reg[1])
            except mysql.connector.Error as error:
                print("Error de BBDD: {}".format(error), "(", self.connData['svr'], ")")
            finally:
                pass
            #self.close()
    ##TODO: capturar errores SQL
        return (status,server)

    '''Leer info de cámaras de BD
        Output: <class 'list'> [{
        'nombre': 'cam1', 'minUbicacion': [ANCHOpx, ALTOpx], 
        'ip': '192.168.0.10', 'desc': 'camara del techo', 
        'ubicaciones': [
          {'nro': 1, 'coord': [X1, Y1], 'size'}, 
          {'nro': 2, 'coord': [X2, Y2]}, 
        ]}] '''
    def readCamInfo(self):
        if self.connect():
            try:
                self.cursor.execute("SELECT config FROM camara WHERE activa = true")
                reg = self.cursor.fetchall()
                cams = list()
                if not reg is None:
                    for r in reg:
                        cams.append(json.loads(r[0]))
            except mysql.connector.Error as error:
                print("Error de BBDD: {}".format(error), "(", self.connData['svr'], ")")
            finally:
                pass
            #self.close()
    ##TODO: capturar errores SQL
        return cams

    def writeCamInfo(self,cams=[]):
        if self.connect():
            try:
                for cam in cams:
                    self.cursor.execute("""UPDATE camara SET 
                                        config = %s 
                                        WHERE nombre = %s and activa = true""",
                                        (json.dumps(cam),cam["nombre"],))
            ##TODO: capturar errores SQL
                self.conn.commit()
            except mysql.connector.Error as error:
                print("Error de BBDD: {}".format(error), "(", self.connData['svr'], ")")
            finally:
                pass 

    '''Leer info de ocupación de ubicaciones de BBDD
        Output: <class 'list'> ['0', '0', '0'] '''
    def readOcupyState(self):
        if self.connect():
            try:
                self.cursor.execute("""SELECT estadoUbicaciones FROM estado 
                                    WHERE camserver = %s 
                                    ORDER BY tstamp DESC LIMIT 1""", 
                                    (self.camsvr.nombre,))
                reg = self.cursor.fetchone()
                estado = list()
                if not reg is None:
                    estado = wrap(reg[0],1)
            ##TODO: capturar errores SQL
            except mysql.connector.Error as error:
                print("Error de BBDD: {}".format(error), "(", self.connData['svr'], ")")
            finally:
                pass
        return estado
        
    def writeOcupyState(self,tnewState=None, newState=[]):
        newState = ''.join(newState)
        if newState != "":
            if self.connect():
                try:
                    self.cursor.execute("""INSERT INTO estado (tstamp, prioridad, estadoUbicaciones, camserver) 
                                                VALUES (%s, 255, %s, %s)
                                                ON DUPLICATE KEY UPDATE tstamp=VALUES(tstamp), estadoUbicaciones=VALUES(estadoUbicaciones)""", 
                                                (tnewState, newState, self.camsvr.nombre))
                    print("ESTADO UBICACIONES: ",newState)
                    # script = self.cursor.execute("""UPDATE estado SET tstamp=%s, estadoUbicaciones=%s 
                    #                             WHERE camserver = %s""", 
                    #                             (tnewState, newState, self.camsvr.nombre))
            ##TODO: capturar errores SQL
                    self.conn.commit()
                except mysql.connector.Error as error:
                    print("Error de BBDD: {}".format(error), "(", self.connData['svr'], ")")
                except BaseException as e:
                    print(time.now(), "Error desconocido grabando estado: ", e)
                finally:
                    pass
       
    def close(self):
        if self.conn and self.conn.is_connected():
            self.cursor.close()
            self.conn.close()
    ##TODO: capturar errores SQL
    
#import datetime
#query = ("SELECT first_name, last_name, hire_date FROM employees "
#         "WHERE hire_date BETWEEN %s AND %s")
#
#hire_start = datetime.date(1999, 1, 1)
#hire_end = datetime.date(1999, 12, 31)
#
#cursor.execute(query, (hire_start, hire_end))
#
#for (first_name, last_name, hire_date) in cursor:
#  print("{}, {} was hired on {:%d %b %Y}".format(
#    last_name, first_name, hire_date))
