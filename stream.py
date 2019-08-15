#https://gist.github.com/keithweaver/5bd13f27e2cc4c4b32f9c618fe0a7ee5

from object_detection.utils import visualization_utils as vis_util
import numpy as np
class CamStream():
    def __init__(self):
        self.cams = None
        self.stream = {}
    
    def setup(self, cams):
        if not self.cams is None:
            # Detener streams en emisión de cámaras no configuradas
            for k, s in self.stream:
                if not k in cams.cams:
                    self.stopStream(k)
        self.cams = cams
    
    def startStream(self, cam, image_np, boxes, classes, scores, categIdx):
        print('Start stream', cam)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        categIdx,
        max_boxes_to_draw=200, 
        use_normalized_coordinates=True,
        line_thickness=8)
        cv2.imshow('object detection', image_np)

    def stopStream(self, cam):
        print('Stop stream', cam)
        pass