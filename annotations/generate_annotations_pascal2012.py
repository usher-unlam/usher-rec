# enconding: utf-8
import os
import gc
import cv2
import psutil
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from annotations.classes.box import Box
from annotations.classes.queryvideo import QueryVideo
from annotations.classes.queryimage import QueryImage


VIDEO_EXTENSION = ['.mov', '.avi', '.mpeg', '.mp4']
VIDEO_ROTATE = 1
VIDEO_STEP = 90

PATH_IN_IMAGES = "\IN"
PATH_OUT_IMAGES =  "\OUT"
NAME_XML = "frame0.xml"

PATH_OUT = '\OUT'
JPG_VOC2012_OUT = PATH_OUT + '\JPEGImages\JPEGImages'
XML_VOC2012_OUT = PATH_OUT + '\VOC2012\Annotations'

UNKNOWN = "Unknown"
UNSPECIFIED = "Unspecified"
PRODUCT = "product"
MAIN_IMAGE_FOLDER = "JPEGImages"

def replicateAnnotations(videos_dbb, annotations):
    import pdb; pdb.set_trace()
    videos_db = iter(videos_dbb)
    while True:
        video = video_db.__next__()
        import pdb; pdb.set_trace()
        for current_frame in video.frames:
            if (count % VIDEO_STEP == 0):
                image = QueryImage(current_frame)
                copyfile(src, dst)
                save_xml(image, annotations)
        
    return nameImages_bdd


def open_videos_frames(path_in, path_out, filetype_list, rotate):
    videos_db = []
    import pdb; pdb.set_trace()
    for root, subdirs, files in os.walk(path_in):
        import pdb; pdb.set_trace()
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
    import pdb; pdb.set_trace()
    return videos_db

def main():
    videos_db = open_videos_frames(PATH_IN_IMAGES, PATH_OUT_IMAGES, VIDEO_EXTENSION, VIDEO_ROTATE)
    
    anotation = ET.parse('frame0.xml')
    anotation.getroot()
    replicateAnnotations(videos_db, anotation)
    
    
if __name__ == '__main__':
    main()
