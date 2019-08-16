import os
import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0
    stream = None

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(os.environ['OPENCV_CAMERA_SOURCE'])
            # Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def __getStream():
        if (Camera.stream is None 
            or not Camera.stream.isOpened()):
            Camera.stream = cv2.VideoCapture(Camera.video_source)
            # Prueba de limitar tamano de frame
            Camera.stream.set(cv2.CAP_PROP_FRAME_WIDTH,640) #3
            Camera.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,480) #4
            Camera.stream.set(cv2.CAP_PROP_FPS,30) # FPS Capturados
            Camera.stream.set(cv2.CAP_PROP_POS_MSEC,6000) #inicia con 6 segundos
        if not Camera.stream.isOpened():
            raise RuntimeError('Could not start camera.')
        return Camera.stream


    @staticmethod
    def frames():
        while True:

            # initiate or reinitiate stream from source
            camera = Camera.__getStream()

            # read current frame
            rec, img = camera.read()

            if img is None:
                # show error image
                img = Camera.error_img
                if not (Camera.stream is None
                    and Camera.stream.isOpened()):
                    Camera.stream.release()
            
            # encode as a jpeg image and return it
            # cv2.imencode(ext, img[, params]) → retval, buf
            # params => https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html#bool%20imwrite(const%20String&%20filename,%20InputArray%20img,%20const%20vector%3Cint%3E&%20params)
            yield cv2.imencode('.jpg', img)[1].tobytes()


