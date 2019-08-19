#https://gist.github.com/keithweaver/5bd13f27e2cc4c4b32f9c618fe0a7ee5

import numpy as np
import datetime
from threading import Thread
from werkzeug.serving import make_server
from object_detection.utils import visualization_utils as vis_util
from flask import Flask, render_template, Response
from video_streaming.camera_opencv import Camera

import json
from flask import jsonify
from conector import Status, CamStatus

#Declaracion de variable global
webserver = None

class defApp:
    def __init__(self,args):
        pass
    def route(self,args):
        pass

class ServerThread(Thread):
    app = defApp("App")
    port = 5000

    def __init__(self, port=5000, app=None):
        Thread.__init__(self,name="StreamServerThread")
        self.port = port
        if app is None:
            self.app = Flask("ServerThread")
        else:
            self.app = app
        #self.app.name = appName
        self.srv = make_server('0.0.0.0',  port, self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()

    
    def clone(self):
        return ServerThread(self.port, self.app)
        
    def run(self):
        print('starting server')
        self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()

    # @app.route('/video_feed')
    # def video_feed():
    #     """Video streaming route. Put this in the src attribute of an img tag."""
    #     return Response(gen(Camera()),
    #                     mimetype='multipart/x-mixed-replace; boundary=frame')


class CamStream():
    app = Flask("CamStream")
    camserver = None
    cams = None

    @staticmethod
    def mycast(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()
        if isinstance(o, CamStatus):
            return "[" + str(int(o)) + "," + str(o) + "]"
 
    @staticmethod
    @app.route('/')
    def index():
        """Video streaming home page."""
        try:
            html = render_template('index.html')
        except:
            html = ('<html><head></head><body><ul>'
                +'<li><a href="/camserver">Estado de Servidor CamServer</a></li>'
                +'<li><a href="/cameras">Estado de Camaras</a></li>'
                +'</ul></body></html>')
        return html

    @staticmethod
    @app.route('/camserver')
    def status():
        """Status of camserver."""
        ret = json.dumps(CamStream.camserver.getStatus(), default=CamStream.mycast)
        return Response(response=ret,
                        status=200,
                        mimetype="application/json")
        #return jsonify(CamStream.camserver.getStatus())

    @staticmethod
    @app.route('/cameras')
    def cameras():
        """Status of cameras."""
        ret = json.dumps(CamStream.cams.getStatus(), default=CamStream.mycast)        
        return Response(response=ret,
                        status=200,
                        mimetype="application/json")
        #return jsonify(CamStream.cams.getStatus())


    def __init__(self):
        self.cams = None
        self.camserver = None
        self.stream = {}
    
    def setup(self, camserver, cams):
        CamStream.camserver = camserver
        CamStream.cams = cams
       # if CamStream.cams is None:
       #     # Detener stream
       #     self.stopStream()
       # else:
        self.startStream()
    
    # Inicia webserver para streaming
    def startStream(self):
        print('Start streaming')
        self.__start_server(__class__)
        print('HTTP server started')

    # Detiene webserver 
    def stopStream(self):
        print('Stop streaming')
        self.__stop_server()
        pass

    # Genera una imagen a partir de frame + boxes|diseño + texto|diseño
    @staticmethod
    def getImageWithBoxes(cam, image_np, boxes, classes, scores, categIdx):
        # Visualizar los resultados de la detección
        vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        categIdx,
        max_boxes_to_draw=400, 
        use_normalized_coordinates=True,
        line_thickness=8)

        return image_np
    

    # Genera una imagen a partir de frame + boxes|diseño + texto|diseño y lo publica en webserver
    def sendStream(self,cam, image_np, boxes, classes, scores, categIdx):
        img = CamStream.getImageWithBoxes(cam, image_np, boxes, classes, scores, categIdx)
        cv2.imshow('object detection', img)
    
    def __start_server(self,appName='myapp'):
        global webserver
        if not webserver:
            # self.app = Flask(appName)
            print('HTTP server created')
            ...
            webserver = ServerThread(5000,CamStream.app)
        else:
            if not webserver.is_alive():
                webserver = webserver.clone()
            pass
        webserver.start()
        print('HTTP server starting')

    def __stop_server(self):
        global webserver
        if webserver:
            webserver.shutdown()
            print('HTTP server stopped')
        else:
            print('HTTP server already stopped')
    
    