# coding=UTF-8
import os
import cv2
import time


class QueryVideo:
    def __init__(self, file, path_out, rotate):
        self.dirname = os.path.dirname(file)
        self.basename = os.path.splitext(os.path.basename(file))[0]
        self.extension = os.path.splitext(file)[1]
        self.save_frames_path = path_out + '/' + self.basename

        self.stream = cv2.VideoCapture(file)
        self.frame_rate = int(self.stream.get(cv2.CAP_PROP_FPS))
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.time_started = time.time()
        self.frames = []

    def finished(self):
        self.time_finished = time.time()

    def total_time(self):
        return self.time_finished - self.time_started

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width
