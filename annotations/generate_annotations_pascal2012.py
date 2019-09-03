# enconding: utf-8
import os
import cv2
import sys
from shutil import copyfile
sys.path.append("..")
from annotations.classes.queryvideo import QueryVideo
from annotations.classes.queryimage import QueryImage

VIDEO_EXTENSION = ['.mov', '.avi', '.mpeg', '.mp4']
VIDEO_ROTATE = 0
VIDEO_STEP = 45
# EXAMPLE GRAY "cv2.COLOR_BGR2GRAY"
# EXAMPLE HSV "cv2.COLOR_BGR2HSV"
# EXAMPLE RGB ""
COLOR = "cv2.COLOR_BGR2HSV"

PATH_IN_IMAGES = "D:/proyecto/usher-rec/annotations/IN/"
PATH_OUT_IMAGES = "D:/proyecto/usher-rec/annotations/OUT/"
NAME_XML = "D:/proyecto/usher-rec/annotations/frame0.xml"

JPG_VOC2012_OUT = PATH_OUT_IMAGES + 'JPEGImages/JPEGImages/'
XML_VOC2012_OUT = PATH_OUT_IMAGES + 'VOC2012/Annotations/'


def replicateAnnotations(videos_dbb, annotation, xml_out):
    if not os.path.exists(xml_out):
        os.makedirs(xml_out)
    video_db = iter(videos_dbb)
    while True:
        video = video_db.__next__()
        count_frame = 0
        for current_frame in video.frames:
            if (count_frame % VIDEO_STEP == 0):
                image = QueryImage(current_frame)
                try:
                    copyfile(annotation, xml_out + '/' + image.basename + ".XML")
                    print("Copy: " + image.basename + ".XML")
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                    exit(1)
                except:
                    print("Unexpected error:", sys.exc_info())
                    exit(1)
            count_frame += 1


def open_videos_frames(path_in, path_out, filetype_list, rotate):
    videos_db = []
    for root, subdirs, files in os.walk(path_in):
        for file in files:
            if os.path.splitext(file)[1].lower() in filetype_list:
                video = QueryVideo(os.path.join(root, file), path_out, rotate)
                if not os.path.exists(JPG_VOC2012_OUT):
                    os.makedirs(JPG_VOC2012_OUT)
                count_frame = 0
                while (video.stream.isOpened()):
                    ret, frame = video.stream.read()
                    if (count_frame % VIDEO_STEP == 0):
                        if not ret:
                            break
                        if rotate:
                            frame = cv2.transpose(frame, frame)
                            frame = cv2.flip(frame, 1)
                        if not COLOR is "":
                            frame = cv2.cvtColor(frame, COLOR)
                        frame_name = JPG_VOC2012_OUT + video.basename + '_' + str('{0:0{width}}'.format(count_frame, width=10)) + '.jpg'
                        cv2.imwrite(frame_name, frame)
                        print("Create frame : " + os.path.splitext(os.path.basename(frame_name))[0])
                    count_frame += 1
                    # Insert connection between videos and frames
                    video.frames.append(frame_name)
                video.stream.release()
            videos_db.append(video)

    if len(videos_db) == 0:
        print('File not exist')
    return videos_db


def main():
    videos_db = open_videos_frames(PATH_IN_IMAGES, PATH_OUT_IMAGES, VIDEO_EXTENSION, VIDEO_ROTATE)
    replicateAnnotations(videos_db, NAME_XML, XML_VOC2012_OUT)


if __name__ == '__main__':
    main()
