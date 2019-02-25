#
# Tello Python3 Control Demo
#
# http://www.ryzerobotics.com/
#
# 1/1/2018


import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2
import time
from time import sleep

import os
import random
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from object_detection.utils import label_map_util, visualization_utils
import glob
from six.moves import urllib
import tarfile

import threading

DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
MODEL_NAME = "C:\\Users\\Hyungi\\Desktop\\ssd_mobilenet_v1_coco_2017_11_17"
PATH_TO_OBJECT_DETECTION_REPO = "C:\\Users\\Hyungi\\Desktop\\tensorflow\\models\\research\\object_detection\\"  # Insert path to tensorflow object detection repository - models/research/object_detection/
PATH_TO_LABELS = PATH_TO_OBJECT_DETECTION_REPO + "data/mscoco_label_map.pbtxt"
NUM_CLASSES = 1

threshold = 0.5

def handler(event, sender, data, **args):
    drone = sender
    # if event is drone.EVENT_FLIGHT_DATA:
        # print(data)

def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[1] - i, coordinates[0] - i)
        rect_end = (coordinates[3] + i, coordinates[2] + i)
        draw.rectangle((rect_start, rect_end), outline = color)

def droneControl():
    global drone
    while True:
        key = input()
        if key == 'q':
            drone.down(50)
        if key == 'w':
            drone.forward()
        if key == 's':
            drone.backward(20)
        if key == 'a':
            drone.left(10)
        if key == 'd':
            drone.right(10)
        if key == 't':
            drone.forward(0)
        if key == 'l':
            drone.land()



class ObjectDetectionPredict():
    """class method to Load tf graph and
    make prediction on test images using predict function
    """

    def __init__(self, model_name):
        """ Downloads, initialize the tf model graph and stores in Memory
        for prediction
        """

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.load_graph(model_name)

    def load_graph(self, model_name):
        """ Loads the model into RAM
        Args:
        model_name: name of model to load
        """
        model_file = model_name + '/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with self.detection_graph.as_default():
            tf.import_graph_def(graph_def)

        self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_operation_by_name('import/image_tensor')
        self.boxes = self.detection_graph.get_operation_by_name('import/detection_boxes')
        self.scores = self.detection_graph.get_operation_by_name('import/detection_scores')
        self.classes = self.detection_graph.get_operation_by_name('import/detection_classes')
        self.num_detections = self.detection_graph.get_operation_by_name('import/num_detections')
        return 0


    def predict_single_image(self, image_file_path):
        """make object detection prediction on single image
        Args:
        image_file_path: Full path of image file to predict
        """
        image_frame = image_file_path

        im_width, im_height = image_frame.shape[:2]
        # print("original_image_frame: ", image_frame)
        # print("original_im_width: ", im_width, "original_im_height: ", im_height )



        image_frame = cv2.resize(image_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        im_width, im_height = image_frame.shape[:2]
        # print("resized_image_frame: ", image_frame)
        # print("resized_im_width: ", im_width, "resized_im_height:" ,im_height )

        #지움 (int(im_width/2), int(im_height/2)))
        # print(image_frame.shape)
        print("image_frame[:, 0:3]: ",image_frame[:, 0:3])
        image_np = image_frame[:, 0:3].reshape((int(im_height), int(im_width), 3)).astype(np.uint8)#테스트
        # image_np = np.array(image_frame.getdata())[:,0:3].reshape((int(im_height), int(im_width), 3)).astype(np.uint8)
        #
        #
        image_np_expanded = np.expand_dims(image_np, axis=0)#테스트
        print("image_np_expanded:", image_np_expanded)
        print("image_np_expanded.shape:", image_np_expanded)
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        boxes, scores, classes, num_detections = 0, 0, 0, 0#임시로 준 값. 지워도 됨
        # (boxes, scores, classes, num_detections) = self.sess.run(
        #     [self.boxes.outputs[0], self.scores.outputs[0], self.classes.outputs[0], self.num_detections.outputs[0]],
        #     feed_dict={self.image_tensor.outputs[0]: image_np_expanded})


        # correct_prediction = [(s, np.multiply(b, [im_height, im_width, im_height, im_width]), c)
        #                                             for c, s, b in zip(classes[0], scores[0], boxes[0]) if (s > threshold and c in self.category_index)]
        # if correct_prediction:
        #     scores, boxes, classes = zip(*correct_prediction)
        #     draw = ImageDraw.Draw(image_frame)
        #     for s, b, c in correct_prediction:
        #         draw_rectangle(draw, b, 'red', 5)
        # else:
        #     scores, boxes, classes = [], [], []

        # print("Number of detections: {}".format(len(scores)))
        # print("\n".join("{0:<20s}: {1:.1f}%".format(self.category_index[c]['name'], s*100.) for (c, s, box) in zip(classes, scores, boxes)))
        return scores, classes, image_frame, boxes


drone = tellopy.Tello()
def main():
    global drone
    prediction_class = ObjectDetectionPredict(model_name=MODEL_NAME)

    try:
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
        drone.connect()
        drone.wait_for_connection(60.0)

        container = av.open(drone.get_video_stream())
        # skip first 300 frames
        frame_skip = 1000
        bbox = (287, 23, 86, 320)
        c = 0

        # drone.takeoff()
        # sleep(5)

        th = threading.Thread(target=droneControl)
        th.start()

        landflag = False
        while True:
            print("******************\nNEW WHILE\n******************** ")
            for frame in container.decode(video=0):

                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue

                if c < 2 :
                    c += 1
                    continue
                else :
                    c = 0

                pil_im = np.array(frame.to_image())
                print("pil_im :", pil_im)

                timer = cv2.getTickCount()


                scores, classes, img, boxes = prediction_class.predict_single_image(pil_im)
                opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                print('FPS : ' + str(float(fps)))
                cv2.putText(opencvImage, "FPS : " + str(int(fps)), (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.imshow('opencvImage', opencvImage)



                if cv2.waitKey(1) & 0xFF == ord('q'):
                  landflag = True
                  cv2.destroyAllWindows()
                  break
            if landflag:
                print('down')
                drone.down(50)
                sleep(3)
                drone.land()
                sleep(1)
                break



        # prediction_class.sess.close()

        print('down again')
        drone.land()
        sleep(1)

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
