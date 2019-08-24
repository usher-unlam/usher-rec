import cv2
vidcap = cv2.VideoCapture('Sin título 6.mp4')
success,image = vidcap.read()
count = 0
cv2.imwrite("frame%d.jpg" % count, image)       
