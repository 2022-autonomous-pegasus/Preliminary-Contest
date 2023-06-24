#!/usr/bin/env python
# -*- coding: utf-8 -*-

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, math
import rospy, rospkg, time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# from utils import perspectiveWrap
from xycar_msgs.msg import xycar_motor
from math import *
import signal
import sys
import os

from matplotlib import pyplot as plt






#=============================================
# 터미널에서 Ctrl-C 키입력으로 프로그램 실행을 끝낼 때
# 그 처리시간을 줄이기 위한 함수
#=============================================
def signal_handler(sig, frame):
    import time
    time.sleep(3)
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0]) # 카메라 이미지를 담을 변수
bridge = CvBridge() 
motor = None # 모터 토픽을 담을 변수

#=============================================
# 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30    # 카메라 FPS - 초당 30장의 사진을 보냄
IMG_WIDTH, IMG_HEIGHT = 640, 480    # 카메라 이미지 가로x세로 크기

# ROI
ASSIST_BASE_LINE = 300 # y
ASSIST_BASE_WIDTH = 50 # y_width


# ==================================
# BIRDSEYE VIEW
# ==================================
def perspectiveWrap(img):
    img_size = IMG_WIDTH, IMG_HEIGHT

    # perspective points to be wraped
    src = np.float32([[200, 330], [0, 480], [480, 330], [680, 480]])

    # Window to be shown
    dst = np.float32([[200, 0], [200, 480], [480, 0], [480, 480]])

    # Matrix to wrap the image for birdseye window
    matrix = cv2.getPerspectiveTransform(src, dst)
    birdseye = cv2.warpPerspective(img, matrix, img_size)

    # Inverse Matrix to unwrap the image for final window
    minv = cv2.getPerspectiveTransform(dst, src)

    return birdseye, minv


#=============================================
# ROI
#=============================================
# img에 ROI 설정
def region_of_interest(img, points):
    img_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8) # 
    pt = np.array(points, np.int32)
    cv2.fillPoly(img_mask, [pt], (255, 255, 255)) #
    masked_img = cv2.bitwise_and(img, img_mask)
    return masked_img

# img_line에 ROI 박스 그리기
def draw_ROI(img, points):
    print(points[0], points[1], points[2], points[3])
    # Draw ROI Box
    cv2.line(img, tuple(points[0]), tuple(points[1]), (255, 255, 255), 1, 0)
    cv2.line(img, tuple(points[1]), tuple(points[2]), (255, 255, 255), 1, 0)
    cv2.line(img, tuple(points[2]), tuple(points[3]), (255, 255, 255), 1, 0)
    cv2.line(img, tuple(points[3]), tuple(points[0]), (255, 255, 255), 1, 0)
    return img



#=============================================
# img -> blur -> canny 변환하여 반환
#=============================================
def Canny_Edge_Detection(img):
    blur_img = cv2.blur(img, (3,3)) # 이미지에 low pass filter kernal을 convolution
    canny_img = cv2.Canny(blur_img, 70, 170, 3) # 이미지의 edge detecting
    return canny_img


#=============================================
# line 중심점 -> angle로 반환
#=============================================
def center_to_angle(line_center, gradient, LINE_exist):
    gradient_avg = np.mean(gradient) * -1 # 검출된 차선들의 평균 기울기 구하기

    # 한 쪽 차선만 보일 때 center값이 편향되는 것을 조정.
    if (gradient_avg > 0.3 and line_center < IMG_WIDTH/2) or (LINE_exist[0] == 1 and LINE_exist[1] == 0): # 왼쪽 차선만 보임
        print("left line only!!!")
        line_center = (line_center + IMG_WIDTH) / 2 + gradient_avg * 10
    elif (gradient_avg < -0.3 and line_center > IMG_WIDTH/2) or (LINE_exist[0] == 0 and LINE_exist[1] == 1): # 오른쪽 차선만 보임
        print("right line only!!!")
        line_center = (line_center - 20) / 2  + gradient_avg * 10

    steer_error = line_center - IMG_WIDTH/2 # 이미지 중심점과 center 좌표 간의 차이

    P_gain = 0.15 # gain
    steer_angle = steer_error * P_gain # gain을 곱하여 angle 도출
    print("gradient_avg : " + str(gradient_avg)) # 평균 기울기
    print("steer_error : " + str(steer_error)) # 차의 방향과 center점의 차이
    print("steer_angle : " + str(steer_angle)) # angle
    return steer_angle


#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
# 카메라 이미지 토픽이 도착하면 자동으로 호출되는 함수
# 토픽에서 이미지 정보를 꺼내 image 변수에 옮겨 담음.
#=============================================
def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

#=============================================
# 모터 토픽을 발행하는 함수  
# 입력으로 받은 angle과 speed 값을 
# 모터 토픽에 옮겨 담은 후에 토픽을 발행함.
#=============================================
def drive(angle, speed):

    global motor

    motor_msg = xycar_motor()
    motor_msg.angle = angle
    motor_msg.speed = speed

    motor.publish(motor_msg)

#=============================================
# 실질적인 메인 함수 
# 카메라 토픽을 받아 각종 영상처리와 알고리즘을 통해
# 차선의 위치를 파악한 후에 조향각을 결정하고,
# 최종적으로 모터 토픽을 발행하는 일을 수행함. 
#=============================================
def start():

    # 위에서 선언한 변수를 start() 안에서 사용하고자 함
    global motor, image

    #=========================================
    # ROS 노드를 생성하고 초기화 함.
    # 카메라 토픽을 구독하고 모터 토픽을 발행할 것임을 선언
    #=========================================
    rospy.init_node('driving')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    image_sub = rospy.Subscriber("/usb_cam/image_raw/",Image,img_callback)

    print ("----- Xycar self driving -----")

    # 첫번째 카메라 토픽이 도착할 때까지 기다림.
    while not image.size == (IMG_WIDTH * IMG_HEIGHT * 3):
        continue
    
    # 각도와 속도의 초기값
    angle = 0
    speed = 10

    # ROI
    margin = 100 # x
    points = []
    points.append([margin, ASSIST_BASE_LINE - ASSIST_BASE_WIDTH])
    points.append([margin, ASSIST_BASE_LINE + ASSIST_BASE_WIDTH])
    points.append([IMG_WIDTH-margin, ASSIST_BASE_LINE + ASSIST_BASE_WIDTH])
    points.append([IMG_WIDTH-margin, ASSIST_BASE_LINE - ASSIST_BASE_WIDTH])


 
    #=========================================
    # 메인 루프 
    # 카메라 토픽이 도착하는 주기에 맞춰 한번씩 루프를 돌면서 
    # "이미지처리 +차선위치찾기 +조향각결정 +모터토픽발행" 
    # 작업을 반복적으로 수행함.
    #=========================================
    while not rospy.is_shutdown():

        # 이미지처리를 위해 카메라 원본이미지를 img에 복사 저장
        img = image.copy()  
        
        # 디버깅을 위해 모니터에 이미지를 디스플레이
        cv2.imshow("CAM View", img)
        cv2.waitKey(1)       
       
        #=========================================
        # IMAGE PROCESSING 
        #=========================================
        birdView, minv = perspectiveWrap(img) # birdseye view 적용
        
        # show detected lines
        img_line = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)

        # thresholding
        img_thres = np.copy(birdView) # image 복사

        #  BGR 제한 값 설정
        blue_threshold = 200
        green_threshold = 200
        red_threshold = 200
        bgr_threshold = [blue_threshold, green_threshold, red_threshold]

        # 횡단보도가 houghlinesP 함수에 의해 탐지되는 것을 방지하기 위해
        # BGR 제한 값보다 작으면 검은색으로 만듦.
        thresholds = (birdView[:,:,0] < bgr_threshold[0]) \
                    | (birdView[:,:,1] < bgr_threshold[1]) \
                    | (birdView[:,:,2] < bgr_threshold[2])
        img_thres[thresholds] = [0,0,0]
        cv2.imshow("white", img_thres)

        img_canny_edge = Canny_Edge_Detection(img_thres) # canny
        img_roi_canny_edge = region_of_interest(img_canny_edge, points) # ROI

        #=========================================
        # hough Transform을 사용한 차선 검출
        #=========================================
        linesP = cv2.HoughLinesP(img_roi_canny_edge, 1, math.pi/180, 30, None, 15, 10)

        # 인식된 차선이 없는 경우 angle=0, speed=7
        if linesP is None:
            drive(0, 7)
            continue


        #=========================================
        # y = 300(ROI)일 때 차선과의 교차점 표시
        # 검출한 차선 그리기
        #=========================================
        print(linesP)
        
        gradient = [] # 기울기
        intersect = [] # 절편
        intersect_base = [] # x = b + a*y
        for i in range(len(linesP)):

            L = linesP[i][0].tolist()

            # 횡단보도 등 가로선 제외
            if 30 < L[3] - L[1] < 30:
                continue
            else:
                line_num = len(intersect_base)
            
            gradient.append(((L[2] - L[0])*1.0) / ((L[3] - L[1])*1.0)) # 기울기 x/y
            intersect.append(L[0] - gradient[line_num] * L[1]) # x절편
            intersect_base.append(intersect[line_num] + gradient[line_num] * ASSIST_BASE_LINE) # x = b + a*y
            cv2.circle(img_line, (int(intersect_base[line_num]), ASSIST_BASE_LINE), 5, (255, 0, 0), -1) # 선과 y = 300의 교차점을 원으로 표시
            cv2.line(img_line, (L[0], L[1]), (L[2], L[3]), (255, 255, 255), 4, cv2.LINE_AA) # 차선을 img_line에 표시

        #=========================================
        # FIND CENTER
        # 연결된 픽셀들을 그룹화하여 무게중심을 찾고 중심점을 계산
        #=========================================
        if len(linesP) != 0:
            cnt, _, stats, centroids = cv2.connectedComponentsWithStats(img_line, 8, cv2.CV_32S)
            print("cnt : " + str(cnt)) # 객체 수 + 1
            c_x_sum = 0
            LINE_exist = [0, 0] # ROI의 중심을 기준으로 왼쪽과 오른쪽에 차선 존재 여부를 저장

            # 레이블링 결과에 사각형과 넘버 표시하고 중심점 구하기.
            for i in range(1, cnt): # 각각의 객체 정보에 들어가기 위한 반복문, 범위를 1부터 시작한 이유는 배경을 제외
                # 평균으로 중심점 구하기
                area = stats[i][cv2.CC_STAT_AREA] # 픽셀 수
                c_x = centroids[i][0] # x 무게중심
                c_y = centroids[i][1] # y 무게중심
                print("Centroid" ,area, c_x, c_y)
                c_x_sum = c_x_sum + c_x # 평균내기위해 다 더함.

                if margin <= c_x <= IMG_WIDTH/2: # 왼쪽 차선이 보임
                    LINE_exist[0] = 1
                elif IMG_WIDTH/2 - 50 <= c_x <= IMG_WIDTH - margin: # 오른쪽 차선이 보임
                    LINE_exist[1] = 1
            
            line_center = c_x_sum / (cnt - 1) # 평균
            print("Centroid Center : " + str(line_center))
            cv2.line(img_line, (IMG_WIDTH//2, IMG_HEIGHT), (int(line_center), 330), (255, 255, 255), 4, cv2.LINE_AA) # 차의 위치로부터 center점까지 선으로 연결하여 시각화.


        #=========================================
        # DISPLAY
        #=========================================
        img_line = draw_ROI(img_line, points) # ROI를 박스 형태로 시각화.
        cv2.imshow("ROI Edge Window", img_roi_canny_edge)
        cv2.imshow("Line Image window", img_line)


        #=========================================
        # angle 결정
        #=========================================
        angle = int(center_to_angle(line_center, gradient, LINE_exist)) # center의 x좌표 -> angle

        #=========================================
        # 차량의 속도 값인 speed값 정하기.
        # 직선 코스에서는 빠른 속도로 주행하고 
        # 회전구간에서는 느린 속도로 주행하도록 설정함.
        #=========================================
        if (angle > 5 or angle < -5) and speed >= 7:   # 회전구간에서는 점점 느리게
            speed -= 1
        elif speed <= 15:                           # 직선구간에서는 점점 빠르게
            speed += 1

        print("speed : " + str(speed))
        # drive() 호출. drive()함수 안에서 모터 토픽이 발행됨.
        drive(angle, speed)


#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함.
# start() 함수가 실질적인 메인 함수임. 
#=============================================
if __name__ == '__main__':
    start()