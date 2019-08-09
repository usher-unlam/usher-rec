# enconding: utf-8
import os
import gc
import cv2
import psutil
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
import sys
from classes.box import Box
from classes.queryvideo import QueryVideo
from classes.queryimage import QueryImage
sys.path.append("..")
from knn import box_real_pixel, MemoryBar

THRESHOLD = 0.93

VIDEO_EXTENSION = ['.mov', '.avi', '.mpeg', '.mp4']
VIDEO_ROTATE = 1
VIDEO_STEP = 90

PATH_IN = '/home/german/Workspace/cat-local/ExampleIn'
PATH_OUT = '/home/german/Workspace/cat-local/ExampleOut'
JPG_VOC2012_OUT = PATH_OUT + '/JPEGImages/JPEGImages'
XML_VOC2012_OUT = PATH_OUT + '/VOC2012/Annotations'
PATH_MODEL = '/home/german/Workspace/google/models/research/ml_only_products_first_2019_02_21_170344/frozen_inference_graph.pb'

UNKNOWN = "Unknown"
UNSPECIFIED = "Unspecified"
PRODUCT = "product"
MAIN_IMAGE_FOLDER = "JPEGImages"


def save_xml(image):
    if not os.path.exists(XML_VOC2012_OUT):
        os.makedirs(XML_VOC2012_OUT)

    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = MAIN_IMAGE_FOLDER
    ET.SubElement(annotation, "filename").text = image.basename + image.extension
    ET.SubElement(annotation, "path").text = image.dirname

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = UNKNOWN

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image.width)
    ET.SubElement(size, "height").text = str(image.height)
    ET.SubElement(size, "depth").text = str(3)

    ET.SubElement(annotation, "segmented").text = str(0)
    for box in image.bounding_boxes:
        obj = ET.SubElement(annotation, "object")
        # product is a generic name
        ET.SubElement(obj, "name").text = PRODUCT
        ET.SubElement(obj, "pose").text = UNSPECIFIED
        ET.SubElement(obj, "truncated").text = str(0)
        # Check only THRESHOLD bad
        ET.SubElement(obj, "difficult").text = str(0)
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(box.y_bottom)
        ET.SubElement(bndbox, "ymin").text = str(box.x_left)
        ET.SubElement(bndbox, "xmax").text = str(box.y_top)
        ET.SubElement(bndbox, "ymax").text = str(box.x_right)
    tree = ET.ElementTree(annotation)
    tree.write(XML_VOC2012_OUT + '/' + image.basename + ".xml", encoding="utf-8", xml_declaration=True)


def assign_subelements(element, subelements):
    for key, value in subelements.items():
        ET.SubElement(element, key).text = value
    return element


def open_videos_frames(path_in, path_out, filetype_list, rotate):
    videos_db = []
    for root, subdirs, files in os.walk(path_in):
        memory = psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)
        for file in files:
            if os.path.splitext(file)[1].lower() in filetype_list:
                video = QueryVideo(os.path.join(root, file), path_out, rotate)
                if not os.path.exists(JPG_VOC2012_OUT):
                    os.makedirs(JPG_VOC2012_OUT)
                count_frame = 0
                print('Opening and getting video frames of ' + video.basename + ' ... using ' + str(memory) + ' MB Memory')
                while (video.stream.isOpened()):
                    ret, frame = video.stream.read()
                    if (count_frame % VIDEO_STEP == 0):
                        if not ret:
                            break
                        if rotate:
                            frame = cv2.transpose(frame, frame)
                            frame = cv2.flip(frame, 1)
                        frame_name = JPG_VOC2012_OUT + '/' + video.basename + '_' + str('{0:0{width}}'.format(count_frame, width=10)) + '.jpg'
                        cv2.imwrite(frame_name, frame)
                    count_frame += 1
                    # Insert connection between videos and frames
                    video.frames.append(frame_name)
                video.stream.release()
            videos_db.append(video)
    if len(videos_db) == 0:
        print('File not exist')
    return videos_db


def main():
    # Tensorflow
    detection_graph = tf.Graph()
    video_db = iter(open_videos_frames(PATH_IN, PATH_OUT, VIDEO_EXTENSION, VIDEO_ROTATE))
    # Video Analysis for frames
    while True:
        try:
            video = video_db.__next__()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(PATH_MODEL, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                sess = tf.Session(graph=detection_graph)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                image = None
                print('\n' + video.basename)
                bar = MemoryBar('Detecting and generating XML ... ', max=len(video.frames))
                count = 0
                for current_frame in video.frames:
                    if (count % VIDEO_STEP == 0):
                        image = QueryImage(current_frame)
                        image_np_expanded = np.expand_dims(image.raw_image, axis=0)
                        # Tensorflow Session
                        (boxes, scores, classes_, num_) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
                        for i, box in enumerate(np.squeeze(boxes)):
                            if(np.squeeze(scores)[i] > THRESHOLD):
                                box_in_pixel = box_real_pixel(box, image.get_width(), image.get_height())
                                image.bounding_boxes.append(Box(image, box_in_pixel))
                        # Create XML PASCAL VOC
                        save_xml(image)
                    count += 1
                    bar.next()
                gc.collect()
        except StopIteration:
            break


if __name__ == '__main__':
    main()
