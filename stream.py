#https://gist.github.com/keithweaver/5bd13f27e2cc4c4b32f9c618fe0a7ee5

import numpy as np
import time
import datetime
from threading import Thread
from werkzeug.serving import make_server
from object_detection.utils import visualization_utils as vis_util
from flask import Flask, render_template, Response
from video_streaming.camera_opencv import Camera

import cv2.cv2 as cv2
import copy
import json
from flask import jsonify
from conector import Status, CamStatus

import collections

def drawBoxesLabels(
    image,
    boxes,
    labels,
    thickness=4,
    color='black',
    mask_alpha=0,
    skip_labels=False):
  """Overlay labeled boxes on an image with formatted scores and label names.
  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.
  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    track_ids: a numpy array of shape [N] with unique track ids. If provided,
      color-coding of boxes will be determined by these ids, and not the class
      indices.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection
  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  use_normalized_coordinates = False
  max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    box = tuple(boxes[i].tolist())
    if True:
      if True:
        display_str = ''
        if not skip_labels:
          if True:
            label = ' '.join(labels[i])
            display_str = str(label)
        # if not skip_scores:
        #   if not display_str:
        #     display_str = '{}%'.format(int(100*scores[i]))
        #   else:
        #     display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        # if not skip_track_ids and track_ids is not None:
        #   if not display_str:
        #     display_str = 'ID {}'.format(track_ids[i])
        #   else:
        #     display_str = '{}: ID {}'.format(display_str, track_ids[i])
        box_to_display_str_map[box].append(display_str)
        # if track_ids is not None:
        #   prime_multipler = _get_multiplier_for_color_randomness()
        #   box_to_color_map[box] = STANDARD_COLORS[
        #       (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
        # else:
        #   box_to_color_map[box] = STANDARD_COLORS[
        #       classes[i] % len(STANDARD_COLORS)]
        box_to_color_map[box] = color

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if mask_alpha > 0:
      vis_util.draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color,
          alpha=mask_alpha
      )
    vis_util.draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    # if keypoints is not None:
    #   draw_keypoints_on_image_array(
    #       image,
    #       box_to_keypoints_map[box],
    #       color=color,
    #       radius=line_thickness / 2,
    #       use_normalized_coordinates=use_normalized_coordinates)
  return image


#Declaracion de variable global
webserver = None


class ServerThread(Thread):
    port = 5000

    def __init__(self, port=5000, app=None, template_folder='templates'):
        Thread.__init__(self,name="StreamServerThread")
        self.port = port
        if app is None:
            self.app = Flask("ServerThread",template_folder=template_folder)
        else:
            self.app = app
        #self.app.name = appName
        self.srv = make_server('0.0.0.0',  port, self.app, threaded=True)
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
    app = Flask("CamStream",template_folder='templates')
    camserver = None
    cams = None
    ubis = None
    lastStCam = ""

    def __init__(self,template_folder='templates'):
        self.cams = None
        self.camserver = None
        self.stream = {}
        self.template_folder = template_folder
        self.app.template_folder = self.template_folder
    
    def setup(self, camserver):
        CamStream.camserver = camserver
        CamStream.cams = camserver.cams
        CamStream.ubis = camserver.ubicaciones
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
        # solo iniciar thread si no esta vivo
        if not webserver.is_alive():
            print('HTTP server starting')
            webserver.start()
        else:
            print('HTTP server still running')

    def __stop_server(self):
        global webserver
        if webserver:
            webserver.shutdown()
            print('HTTP server stopped')
        else:
            print('HTTP server already stopped')
  

    @staticmethod
    def mycast(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()
        if isinstance(o, CamStatus):
            return "[" + str(int(o)) + "," + str(o) + "]"
 
    @staticmethod
    @app.route('/')
    def index():
        """Root home page."""
        try:
            html = render_template('index.html')
        except BaseException as ex:
            html = ('<html><head></head><body><ul>'
                +'<li><a href="/camserver">Estado de Servidor CamServer</a></li>'
                +'<li><a href="/cameras">Estado de Camaras</a></li>'
                +'<li><a href="/live">Ver video en vivo</a></li>'
                +'</ul></body></html>')
        return html

    @staticmethod
    @app.route('/live')
    def live():
        """Live Video streaming home page."""
        try:
            html = render_template('live.html')
        except BaseException as ex:
            html = (
'''<html><head></head>
<body>
    <h4>ERROR MOSTRANDO VIDEO EN VIVO</h4>
    <ul>
    <li><a href="/camserver">Estado de Servidor CamServer</a></li>
    <li><a href="/cameras">Estado de Camaras</a></li>
    <li><a href="/live">Ver video en vivo</a></li>
    </ul>
</body></html>''')
        return html
        #return jsonify(CamStream.camserver.getStatus())

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

    @staticmethod
    def gen(cam):
        """Video streaming generator function."""
        #fr=0
        FRM_REPEAT_BEFORE_ERROR = 100
        repeatLast = 0
        lastimg = Camera.error_img
        img = None
        while True:
            if not cam == "":
                img = CamStream.cams.getLastFrame(cam)
            if img is None:
                if repeatLast > FRM_REPEAT_BEFORE_ERROR:
                    img = Camera.error_img
                else:
                    repeatLast += 1
                    img = lastimg
            else:
                repeatLast = 0
                lastimg = img
                try:
                    numUbi = CamStream.ubis.getNumByCam(cam)
                    coordUbi = CamStream.ubis.getCoordByCam(cam)
                    rectUbi = CamStream.ubis.getYxyxByCam(cam)
                    statUbi = CamStream.ubis.getLastStateByCam(cam)
                    rectRN = CamStream.ubis.getLastDetectionByCam(cam)
                    ###casteo a numpy
                    numUbi = np.expand_dims( numUbi, axis=1)
                    coordUbi = np.expand_dims( coordUbi, axis=1)
                    rectUbi = np.array(rectUbi)
                    #statUbi = np.expand_dims(statUbi[cam], axis=1)
                    rectRN = np.array(rectRN)
                    #Unir los datos a mostrar en pantalla
                    #str_list = np.concatenate((numUbi , numUbi),axis=1)
                    str_list = numUbi
                    str_list = str_list.astype(str)
                    str_list = str_list.tolist()
                    
                    drawBoxesLabels(img,rectUbi,str_list, thickness=6, color='Gold', mask_alpha=0,skip_labels=False)
                    #vis_util.draw_bounding_boxes_on_image(img, rectUbi,display_str_list_list=str_list)

                    drawBoxesLabels(img,rectRN,str_list, thickness=3, color='Aqua', mask_alpha=0,skip_labels=True)
                    #vis_util.draw_bounding_boxes_on_image(img, rectRN, color='Aqua',thickness=3)
                except BaseException as e:
                    print("Problema al dibujar sobre imagen:",e)
            #print("Frame",fr)
            #frame = camera.get_frame()
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0)
            #time.sleep(0.001)
            #fr+=1

    @staticmethod
    @app.route('/video_feed')
    @app.route('/video_feed/<cam>')
    def video_feed(cam="!LAST!"):
        """Video streaming route. Put this in the src attribute of an img tag."""
        if cam == "!LAST!":
            camstat = CamStream.cams.camstat.copy()
            if (CamStream.lastStCam == "" or CamStream.lastStCam not in camstat):
                cam = ""
                # Buscar primer camara con estado OK
                for c,st in camstat.items():
                    if st[1] is CamStatus.OK:
                        CamStream.lastStCam = c
                        break
            cam = CamStream.lastStCam
        if cam == "":
            return Response(None,
                            400)
        #return "./" + cam + "/." + jsonify(stCam) 
        return Response(CamStream.gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')
        #return Response(img, mimetype='multipart/x-mixed-replace; boundary=frame')


    