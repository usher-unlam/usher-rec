# coding=UTF-8
import os


class Box:
    def __init__(self, image, box):
        self.x_left = box[0]
        self.y_bottom = box[1]
        self.x_right = box[2]
        self.y_top = box[3]
        self.image_box = image.raw_image[self.x_left:self.x_right, self.y_bottom:self.y_top].copy
        self.height, self.width, self.channels = self.image_box().shape

    def set_upc(self, name):
        self.path_product_detected = name
        self.upc_detected = self.path_product_detected.split(os.sep)[6].split('__')[0]

    def set_name(self, name):
        self.product_detected = name

    def get_name(self):
        return self.product_detected

    def get_upc(self):
        return self.upc_detected

    def get_path_product_detected(self):
        return self.path_product_detected

    def get_box(self):
        return (self.x_left, self.y_bottom, self.x_right, self.y_top)

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def get_channels(self):
        return self.channels
