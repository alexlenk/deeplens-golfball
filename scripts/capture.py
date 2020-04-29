import cv2
import uuid
import sys
import os
import time

os.system('aws s3 cp ./capture.py s3://ale-golfball-train')
print ("Tools uploaded")


label = str(sys.argv[1])
folder = '/media/aws_cam/700E-6000/' + label + '/'

if len(sys.argv) < 3:
    count = 1
else:
    count = int(sys.argv[2])

if len(sys.argv) < 4:
    rounds = 1
else:
    rounds = int(sys.argv[3])

if len(sys.argv) < 5:
    init_sleep = 10
else:
    init_sleep = int(sys.argv[4])



try:
    os.stat(folder)
except:
    os.mkdir(folder)

cam = cv2.VideoCapture("/opt/awscam/out/ch2_out.mjpeg")


for x in range(rounds):
    print("Get Ready for Round " + str(x) + " (" + str(init_sleep) + " Sek; " + str(count) + " Pics)")
    for z in range(init_sleep):
        print (str(init_sleep - z))
        time.sleep(1)
    print ("GOOOOOOOOOOOOOO")
    time.sleep(1)
    for y in range(count):
        success, frame = cam.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(folder + uuid.uuid4().hex + '.jpeg', frame)
        print('Round ' + str(x + 1) + '; Picure ' + str(y + 1) + ' written')
        time.sleep(0.2)

os.system('aws s3 sync ' + folder + ' s3://ale-golfball-train/' + label)
