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
import numpy
import time


def main():
    tracker = cv2.TrackerKCF_create()
    drone = tellopy.Tello()

    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        container = av.open(drone.get_video_stream())
        # skip first 300 frames
        frame_skip = 300

        bbox = (287, 23, 86, 320)


        roiflag = False
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                if frame_skip == 0 and roiflag == False:
                    image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                    # Uncomment the line below to select a different bounding box
                    bbox = cv2.selectROI(image, False)
                    # Initialize tracker with first frame and bounding box
                    ok = tracker.init(image, bbox)
                    roiflag  = True

                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)


                timer = cv2.getTickCount()

                # Update tracker
                ok, bbox = tracker.update(image)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

                # Draw bounding box
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(image, p1, p2, (255,0,0), 2, 1)
                else :
                    # Tracking failure
                    cv2.putText(image, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

                # Display tracker type on frame
                cv2.putText(image, 'KCF' + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

                # Display FPS on frame
                cv2.putText(image, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);


                cv2.imshow('Original', image)
                # cv2.imshow('Canny', cv2.Canny(image, 100, 200))

                print('before waitkey')
                k = cv2.waitKey(1) & 0xff
                if k == 27 : break
                print('after waitkey')

                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)


    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()







#
# import threading
# import socket
# import sys
# import time
# import platform
#
# host = ''
# port = 9000
# locaddr = (host,port)
#
#
# # Create a UDP socket
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#
# tello_address = ('192.168.10.1', 8889)
#
# sock.bind(locaddr)
#
# def recv():
#     count = 0
#     while True:
#         try:
#             data, server = sock.recvfrom(1518)
#             print(data.decode(encoding="utf-8"))
#         except Exception:
#             print ('\nExit . . .\n')
#             break
#
#
# print ('\r\n\r\nTello Python3 Demo.\r\n')
#
# print ('Tello: command takeoff land flip forward back left right \r\n       up down cw ccw speed speed?\r\n')
#
# print ('end -- quit demo.\r\n')
#
#
# #recvThread create
# recvThread = threading.Thread(target=recv)
# recvThread.start()
#
# while True:
#     try:
#         python_version = str(platform.python_version())
#         version_init_num = int(python_version.partition('.')[0])
#        # print (version_init_num)
#         if version_init_num == 3:
#             msg = input("");
#         elif version_init_num == 2:
#             msg = raw_input("");
#
#         if not msg:
#             break
#
#         if 'end' in msg:
#             print ('...')
#             sock.close()
#             break
#
#         # Send data
#         msg = msg.encode(encoding="utf-8")
#         sent = sock.sendto(msg, tello_address)
#     except KeyboardInterrupt:
#         print ('\n . . .\n')
#         sock.close()
#         break
