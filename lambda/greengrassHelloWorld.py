
#*****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
#*****************************************************
""" A sample lambda for object detection"""
from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import greengrasssdk
import mo
import datetime

class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """
    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()

def greengrass_infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
        thing_name = os.environ['AWS_IOT_THING_NAME']
    except Exception as ex:
        thing_name = "deeplens_cIqzAO10SVWwwHcsqXD5Jg"

    iot_topic = '$aws/things/{}/infer'.format(thing_name)
    if True: #try:
        # This object detection model is implemented as single shot detector (ssd), since
        # the number of labels is small we create a dictionary that will help us convert
        # the machine labels to human readable labels.
        model_type = 'ssd'
        output_map = {0: 'golf ball', 1: 'hole'}
        # Create an IoT client for sending to messages to the cloud.
        client = greengrasssdk.client('iot-data')
        #if os.environ['AWS_IOT_THING_NAME'] == "":
        #    iot_topic = '$aws/things/deeplens_cIqzAO10SVWwwHcsqXD5Jg/infer'
        #else:
        #    iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()
        # The height and width of the training set images
        #input_height = 720

        base_width = 850
        #input_width = 212
        input_width = 150

        input_height = input_width
        scale = base_width / float(input_width)
#        model_algo = "vgg16_reduced"
        model_algo = "resnet50"
        if os.path.isfile('/opt/awscam/artifacts/deploy_ssd_' + model_algo + '_' + str(input_width) + '.xml'):
            model_path = '/opt/awscam/artifacts/deploy_ssd_' + model_algo + '_' + str(input_width) + '.xml'
        else:
            #ret, model_path = mo.optimize('deploy_ssd_' + model_algo + '_' + str(input_width),input_width, input_height, aux_inputs={'--img-channels': 2})
            ret, model_path = mo.optimize('deploy_ssd_' + model_algo + '_' + str(input_width),input_width, input_height)
        #model_path = '/opt/awscam/artifacts/mxnet_deploy_ssd_resnet50_300_FP16_FUSED.xml'
        # Load the model onto the GPU.
        client.publish(topic=iot_topic, payload='Loading object detection model')
        model = awscam.Model(model_path, {'GPU': 1})
        client.publish(topic=iot_topic, payload='Object detection model loaded')
        # Set the threshold for detection
        detection_threshold = 0.5
        # Do inference until the lambda is killed.
        ballIn = False
        ballInTime = datetime.datetime(1970,1,1)
        countOverlap = 0
        mod = 1
        send_no = False
        while True:
            print(" ")
            timeNow = datetime.datetime.now()
            timeDiff = timeNow - ballInTime
            if(timeDiff.total_seconds() >= 5):
                ballIn = False
                ballInTime = datetime.datetime(1970,1,1)
                countOverlap = 0

            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            frame_debug = frame
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Resize frame to the same size as the training set.
            frame_resize = frame[670:None, 900:1750]
#	    frame_resize = frame[670:670+450, 900+250:900+250+450]
#            frame_resize_file = "/tmp/frame_resize.jpeg"
            #if not os.path.exists(frame_resize_file):
            #    os.mkfifo(frame_resize_file)
#            cv2.imwrite(frame_resize_file, frame_resize)
#            frame_resize = cv2.imread(frame_resize_file)
#            frame_resize = cv2.imread("/home/aws_cam/test2-small.jpeg")
            frame_resize = cv2.resize(frame_resize, (input_width, input_height))
#            frame_resize = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
#            frame_resize = cv2.cvtColor(frame_resize, cv2.COLOR_GRAY2BGR)
#            frame_resize = cv2.imread("/home/aws_cam/test2-small.jpeg")
            #cv2.imwrite("/tmp/frame" + str(input_width) + ".jpeg", frame_resize)

            # Run the images through the inference engine and parse the results using
            # the parser API, note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a ssd model,
            # a simple API is provided.
            a = datetime.datetime.now()
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(frame_resize))
            b = datetime.datetime.now()
            c = b - a
            print("Inference Time: " + str(c.total_seconds()))
#            print(parsed_inference_results)
            # Compute the scale in order to draw bounding boxes on the full resolution
            # image.
            yoffset = 670
            xoffset = 900
            # Dictionary to be filled with labels and probabilities for MQTT
            cloud_output = {}
            balls = []
            holes = []
            hole = None
            # Get the detected objects and probabilities
            for obj in parsed_inference_results[model_type]:
                if obj['prob'] > detection_threshold:
                    # Add bounding boxes to full resolution frame
                    xmin = int(xoffset + obj['xmin'] * scale)
                    ymin = int(yoffset + obj['ymin'] * scale)
                    xmax = int(xoffset + obj['xmax'] * scale)
                    ymax = int(yoffset + obj['ymax'] * scale)
                    # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                    # for more information about the cv2.rectangle method.
                    # Method signature: image, point1, point2, color, and tickness.
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 10)
#                    cv2.rectangle(frame_resize, (int(obj['xmin']), int(obj['ymin'])), (int(obj['xmax']), int(obj['ymax'])), (255, 165, 20), 10)
                    # Amount to offset the label/probability text above the bounding box.
                    text_offset = 15
                    # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                    # for more information about the cv2.putText method.
                    # Method signature: image, text, origin, font face, font scale, color,
                    # and tickness
                    cv2.putText(frame_debug, "{}: {:.2f}%".format(output_map[obj['label']],
                                                               obj['prob'] * 100),
                                (xmin, ymin-text_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 165, 20), 6)
                    # Store label and probability to send to cloud
                    cloud_output[output_map[obj['label']]] = obj['prob']
                    if obj['label'] == 1 and obj['prob'] > 0.7:
                        holes.append(obj)
                    elif obj['label'] == 0:
                        balls.append(obj)

            for hole_i in holes:
                if hole is None:
                    hole = hole_i
                else:
                    if hole_i['prob'] > hole['prob']:
                        hole = hole_i

            for ball in balls:
                x_overlap = 0
                y_overlap = 0
                ballArea = 10000
                holeArea = 0
                try:
                    x_overlap = min(hole["xmax"], ball["xmax"]) - max(hole["xmin"], ball["xmin"])
                    y_overlap = min(hole["ymax"], ball["ymax"]) - max(hole["ymin"], ball["ymin"])
                    ballArea = (ball["xmax"] - ball["xmin"]) * (ball["ymax"] - ball["ymin"])
                    holeArea = (hole["xmax"] - hole["xmin"]) * (hole["ymax"] - hole["ymin"])
                except Exception as ex:
                    print(ex)
                overlapArea = max(0, max(x_overlap,0) * max(y_overlap,0))
                overlapPerc = overlapArea / ballArea
                print("Overlap " + str(overlapPerc))
                cloud_output["ballArea"] = ballArea
                cloud_output["holeArea"] = holeArea
                cloud_output["overlapPerc"] = overlapPerc

                if overlapPerc >= 0.5 and ballArea < 370:
                    countOverlap += 1

            if(countOverlap >= 1):
                ballInTime = datetime.datetime.now()
                ballIn = True
                countOverlap = 0
                

            if(ballIn):
                cloud_output["scored"] = "yes"
                cv2.putText(frame, "Scored!", (300, 300), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 165, 20), 6)
                cv2.putText(frame_debug, "Scored!", (300, 300), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 165, 20), 6)
            else:
                cloud_output["scored"] = "no"

            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
    #            cv2.imwrite('/tmp/result.jpeg', frame)
    #            cv2.imwrite('/tmp/frame_resize.jpeg', frame_resize)
            # Send results to the cloud
            if (mod % 1) == 0:
                if(ballIn):
                    send_no = False
                    client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
                mod = 1
            else:
                mod = mod + 1
                
            if(not send_no and not ballIn):
                client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
                send_no = True
                
            print(json.dumps(cloud_output))
            d = datetime.datetime.now()
            runtime = d - timeNow
            print("Total Runtime: " + str(runtime.total_seconds()))
#    except Exception as ex:
#        client.publish(topic=iot_topic, payload='Error in object detection lambda: {}'.format(ex))

greengrass_infinite_infer_run()