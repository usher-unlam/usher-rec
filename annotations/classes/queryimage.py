# coding=UTF-8
import os
import cv2
import time


class QueryImage:
    def __init__(self, file):
        self.dirname = os.path.dirname(file)
        self.basename = os.path.splitext(os.path.basename(file))[0]
        self.extension = os.path.splitext(file)[1]
        self.raw_image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        self.height, self.width, self.channels = self.raw_image.shape
        self.time_started = time.time()
        self.bounding_boxes = []

    def get_full_path(self):
        return (self.dirname + '/' + self.basename + self.extension)

    def finished(self):
        self.time_finished = time.time()

    def total_time(self):
        return self.time_finished - self.time_started

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width
