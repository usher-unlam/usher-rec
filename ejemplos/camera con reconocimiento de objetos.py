import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
#ipcamUrl = 0
#ipcamUrl = 'http://admin:admin@192.168.1.33:8081'
#ipcamUrl = 'http://admin:usher@192.168.0.9:8081'
ipcamUrl = 'http://admin:usher@irv.sytes.net:8081'

cap = cv2.VideoCapture(ipcamUrl)

sys.path.append("..")

# Importación del módulo de detección de objetos.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_CKPT = 'modelo_congelado/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('configuracion', 'label_map.pbtxt')

NUM_CLASSES = 90
FRAMES_OMITIDOS = 10 #Análisis en LAN: frames{fluido,delay}= 4{si,>4"} 7{si,<1"} 10{si,~0"}

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = 'img_pruebas'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

IMAGE_SIZE = (12, 8)

contador=0
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    sess = tf.Session()
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    i = FRAMES_OMITIDOS
    while True:
      ret, image_np = cap.read()
      if (i < FRAMES_OMITIDOS):
        i += 1
      else:
        i = 0    
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.      
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

        cv2.imshow('object detection', image_np)
        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
