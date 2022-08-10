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
from xycar_msgs.msg import xycar_motor
from math import *
import signal
import sys
import os
import random


# colors
red, green, blue, yellow = (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)

class BEV(object):
    '''
    Calibrates camera images to remove distortion and transforms to bird-eye-view image
    ''' 

    def __init__(self):

        # calibration config
        self.img_size = (640, 480)
        self.warp_img_w, self.warp_img_h, self.warp_img_mid = 650, 120, 60

        self.mtx = np.array([[363.090103, 0.000000, 313.080058],
                             [0.000000, 364.868860, 252.739984],
                             [0.000000, 0.000000, 1.000000]])
        self.dist = np.array([-0.334146, 0.099765, -0.000050, 0.001451, 0.000000])
        self.cal_mtx, self.cal_roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, self.img_size, 1, self.img_size)

        # perspective config
        warpx_mid, warpx_margin_hi, warpx_margin_lo, warpy_hi, warpy_lo, tilt = 320, 200, 319, 325, 375, -5
        self.warp_src  = np.array([[warpx_mid+tilt-warpx_margin_hi, warpy_hi], [warpx_mid+tilt+warpx_margin_hi, warpy_hi], 
                                   [warpx_mid-warpx_margin_lo,  warpy_lo], [warpx_mid+warpx_margin_lo, warpy_lo]], dtype=np.float32)
        self.warp_dist = np.array([[100, 0], [649-100, 0],
                                   [100, 119], [649-100, 119]], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(self.warp_src, self.warp_dist)
    
    def to_calibrated(self, img, show=False):
        img = cv2.undistort(img, self.mtx, self.dist, None, self.cal_mtx)
        if show:
            cv2.imshow('calibrated', img)
        return img

    def to_perspective(self, img, show=False):
        img = cv2.warpPerspective(img, self.M, (self.warp_img_w, self.warp_img_h), flags=cv2.INTER_LINEAR)
        if show:
            cv2.imshow('bird-eye-view', img)
        return img

    def __call__(self, img, show_calibrated=False, show=False):
        '''
        return bird-eye-view image of an input image
        '''
        img = self.to_calibrated(img, show=True)
        img = self.to_perspective(img, show=True)
        return img


class LaneDetector():
    '''
    Detects left, middle, right lane from an image and calculate angle of the lane.
    Uses canny, houghlinesP for detecting possible lane candidates.
    Calculates best fitted lane position and predicted lane position from previous result.
    '''

    def __init__(self):

        self.bev = BEV()

        # canny params
        self.canny_low, self.canny_high = 100, 120

        # HoughLineP params
        self.hough_threshold, self.min_length, self.min_gap = 10, 50, 10

        # initial state
        self.angle = 0.0
        self.prev_angle = deque([0.0], maxlen=5)
        self.lane = np.array([90.0, 320., 568.])

        # filtering params:
        self.angle_tolerance = np.radians(30)
        self.cluster_threshold = 25

    def to_canny(self, img, show=False):
        img = cv2.GaussianBlur(img, (7,7), 0)
        img = cv2.Canny(img, self.canny_low, self.canny_high)
        if show:
            cv2.imshow('canny', img)
        return img

    def hough(self, img, show=False):
        lines = cv2.HoughLinesP(img, 1, np.pi/180, self.hough_threshold, self.min_gap, self.min_length)
        if show:
            hough_img = np.zeros((img.shape[0], img.shape[1], 3))
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    cv2.line(hough_img, (x1, y1), (x2, y2), red, 2)
            cv2.imshow('hough', hough_img)
        return lines

    def filter(self, lines, show=True):
        '''
        filter lines that are close to previous angle and calculate its positions
        '''
        thetas, positions = [], []
        if show:
            filter_img = np.zeros((self.bev.warp_img_h, self.bev.warp_img_w, 3))

        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if y1 == y2:
                    continue
                flag = 1 if y1-y2 > 0 else -1
                theta = np.arctan2(flag * (x2-x1), flag * 0.9* (y1-y2))
                if abs(theta - self.angle) < self.angle_tolerance:
                    position = float((x2-x1)*(self.bev.warp_img_mid-y1))/(y2-y1) + x1
                    thetas.append(theta)
                    positions.append(position) 
                    if show:
                        cv2.line(filter_img, (x1, y1), (x2, y2), red, 2)

        self.prev_angle.append(self.angle)
        if thetas:
            self.angle = np.mean(thetas)
        if show:
            cv2.imshow('filtered lines', filter_img)
        return positions

    def get_cluster(self, positions):
        '''
        group positions that are close to each other
        '''
        clusters = []
        for position in positions:
            if 0 <= position < self.bev.warp_img_w:
                for cluster in clusters:
                    if abs(cluster[0] - position) < self.cluster_threshold:
                        cluster.append(position)
                        break
                else:
                    clusters.append([position])
        lane_candidates = [np.mean(cluster) for cluster in clusters]
        return lane_candidates

    def predict_lane(self):
        '''
        predicts lane positions from previous lane positions and angles
        '''
        predicted_lane = self.lane[1] + [-220/max(np.cos(self.angle), 0.75), 0, 240/max(np.cos(self.angle), 0.75)]
        predicted_lane = predicted_lane + (self.angle - np.mean(self.prev_angle))*70
        return predicted_lane

    def update_lane(self, lane_candidates, predicted_lane):
        '''
        calculate lane position using best fitted lane and predicted lane
        '''

        if not lane_candidates:
            self.lane = predicted_lane
            return

        possibles = []

        for lc in lane_candidates:

            idx = np.argmin(abs(self.lane-lc))

            if idx == 0:
                estimated_lane = [lc, lc + 220/max(np.cos(self.angle), 0.75), lc + 460/max(np.cos(self.angle), 0.75)]
                lc2_candidate, lc3_candidate = [], []
                for lc2 in lane_candidates:
                    if abs(lc2-estimated_lane[1]) < 50 :
                        lc2_candidate.append(lc2)
                for lc3 in lane_candidates:
                    if abs(lc3-estimated_lane[2]) < 50 :
                        lc3_candidate.append(lc3)
                if not lc2_candidate:
                    lc2_candidate.append(lc + 220/max(np.cos(self.angle), 0.75))
                if not lc3_candidate:
                    lc3_candidate.append(lc + 460/max(np.cos(self.angle), 0.75))
                for lc2 in lc2_candidate:
                    for lc3 in lc3_candidate:
                        possibles.append([lc, lc2, lc3])

            elif idx == 1:
                estimated_lane = [lc - 220/max(np.cos(self.angle), 0.75), lc, lc + 240/max(np.cos(self.angle), 0.75)]
                lc1_candidate, lc3_candidate = [], []
                for lc1 in lane_candidates:
                    if abs(lc1-estimated_lane[0]) < 50 :
                        lc1_candidate.append(lc1)
                for lc3 in lane_candidates:
                    if abs(lc3-estimated_lane[2]) < 50 :
                        lc3_candidate.append(lc3)
                if not lc1_candidate:
                    lc1_candidate.append(estimated_lane[0])
                if not lc3_candidate:
                    lc3_candidate.append(estimated_lane[2])
                for lc1 in lc1_candidate:
                    for lc3 in lc3_candidate:
                        possibles.append([lc1, lc, lc3])

            else :
                estimated_lane = [lc - 460/max(np.cos(self.angle), 0.75), lc - 240/max(np.cos(self.angle), 0.75), lc]
                lc1_candidate, lc2_candidate = [], []
                for lc1 in lane_candidates:
                    if abs(lc1-estimated_lane[0]) < 50 :
                        lc1_candidate.append(lc1)
                for lc2 in lane_candidates:
                    if abs(lc2-estimated_lane[1]) < 50 :
                        lc2_candidate.append(lc2)
                if not lc1_candidate:
                    lc1_candidate.append(estimated_lane[0])
                if not lc2_candidate:
                    lc2_candidate.append(estimated_lane[1])
                for lc1 in lc1_candidate:
                    for lc2 in lc2_candidate:
                        possibles.append([lc1, lc2, lc])

        possibles = np.array(possibles)
        error = np.sum((possibles-predicted_lane)**2, axis=1)
        best = possibles[np.argmin(error)]
        self.lane = 0.4 * best + 0.6 * predicted_lane

    def mark_lane(self, img, lane=None):
        '''
        mark calculated lane position to an image 
        '''
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if lane is None:
            lane = self.lane
        l1, l2, l3 = self.lane
        cv2.circle(img, (int(l1), self.bev.warp_img_mid), 3, red, 5, cv2.FILLED)
        cv2.circle(img, (int(l2), self.bev.warp_img_mid), 3, green, 5, cv2.FILLED)
        cv2.circle(img, (int(l3), self.bev.warp_img_mid), 3, blue, 5, cv2.FILLED)
        cv2.imshow('marked', img)

    def __call__(self, img, target_lane):
        '''
        returns angle and cte of a target lane from an image
        angle : radians
        cte : pixels
        '''
        canny = self.to_canny(img, show=True)
        bev = self.bev(canny, show=True)
        lines = self.hough(bev, show=True)
        positions = self.filter(lines, show=True)
        lane_candidates = self.get_cluster(positions)
        predicted_lane = self.predict_lane()
        self.update_lane(lane_candidates, predicted_lane)
        self.mark_lane(bev)

        if target_lane == 'middle':
            return self.angle, self.lane[1]
        elif target_lane == 'left':
            return self.angle, self.lane[0]*0.75+self.lane[1]*0.25
        else:
            return self.angle, self.lane[2]*0.75+self.lane[1]*0.25

#=============================================
# 터미널에서 Ctrl-C 키입력으로 프로그램 실행을 끝낼 때
# 그 처리시간을 줄이기 위한 함수
#=============================================
def draw_ROI(img, points):
    print(points[0], points[1], points[2], points[3])
    # Draw ROI Box
    cv2.line(img, tuple(points[0]), tuple(points[1]), (255, 255, 255), 1, 0)
    cv2.line(img, tuple(points[1]), tuple(points[2]), (255, 255, 255), 1, 0)
    cv2.line(img, tuple(points[2]), tuple(points[3]), (255, 255, 255), 1, 0)
    cv2.line(img, tuple(points[3]), tuple(points[0]), (255, 255, 255), 1, 0)
    return img

def view_point_on_img(img, pts):
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i, pt in enumerate(pts):
        pt = tuple(map(int, pt))
        print(pt, type(pt))
        cv2.circle(img, pt, 7, color[i], -1)
    return img

#=============================================
def signal_handler(sig, frame):
    import time
    time.sleep(3)
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def region_of_interest(img, points):
    img_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8) # 
    pt = np.array(points, np.int32)
    cv2.fillPoly(img_mask, [pt], (255, 255, 255)) #
    masked_img = cv2.bitwise_and(img, img_mask)
    return masked_img

def center_to_angle(line_center, gradient, LINE_exist):
    gradient_avg = np.mean(gradient) * -1 # 검출된 차선들의 평균 기울기 구하기

    # 한 쪽 차선만 보일 때 center값이 편향되는 것을 조정.
    # gradient and 
    if (gradient_avg > 0.3 and line_center < IMG_WIDTH/2) or (LINE_exist[0] == 1 and LINE_exist[1] == 0): # 왼쪽 차선만 보임
        print("left line only!!!")
        line_center = (line_center + IMG_WIDTH) / 2 * gradient_avg # (line_center + IMG_WIDTH) / 2 + gradient_avg * 10
    elif (gradient_avg < -0.3 and line_center > IMG_WIDTH/2) or (LINE_exist[0] == 0 and LINE_exist[1] == 1): # 오른쪽 차선만 보임
        print("right line only!!!")
        line_center = (line_center - 20) / 2 * gradient_avg  # (line_center - 20) / 2  + gradient_avg * 10

    steer_error = line_center - IMG_WIDTH/2 # 이미지 중심점과 center 좌표 간의 차이

    P_gain = 0.15 # gain
    steer_angle = steer_error * P_gain # gain을 곱하여 angle 도출
    print("gradient_avg : " + str(gradient_avg)) # 평균 기울기
    print("steer_error : " + str(steer_error)) # 차의 방향과 center점의 차이
    print("steer_angle : " + str(steer_angle)) # angle
    return line_center, int(steer_angle)


#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0]) # 카메라 이미지를 담을 변수
bridge = CvBridge() # OpenCV 함수를 사용하기 위한 브릿지 
motor = None # 모터 토픽을 담을 변수
img_ready = False # 카메라 토픽이 도착했는지의 여부 표시 

#=============================================
# 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30    # 카메라 FPS - 초당 30장의 사진을 보냄
IMG_WIDTH, IMG_HEIGHT = 640, 480    # 카메라 이미지 가로x세로 크기
ROI_ROW = 250   # 차선을 찾을 ROI 영역의 시작 Row값 
ROI_HEIGHT = IMG_HEIGHT - ROI_ROW   # ROI 영역의 세로 크기  
L_ROW = ROI_HEIGHT - 120  # 차선의 위치를 찾기 위한 기준선(수평선)의 Row값
ASSIST_BASE_LINE = 300 # y
ASSIST_BASE_WIDTH = 50 # y_width
#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
# 카메라 이미지 토픽이 도착하면 자동으로 호출되는 함수
# 토픽에서 이미지 정보를 꺼내 image 라는 변수에 옮겨 담음.
# 카메라 토픽의 도착을 표시하는 img_ready 값을 True로 바꿈.
#=============================================
def img_callback(data):
    global image, img_ready
    image = bridge.imgmsg_to_cv2(data, "bgr8")
    img_ready = True
    
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

def start():

    # 위에서 선언한 변수를 start() 안에서 사용하고자 함
    global motor, image, img_ready

    #=========================================
    # ROS 노드를 생성하고 초기화 함.
    # 카메라 토픽을 구독하고 모터 토픽을 발행할 것임을 선언
    #=========================================
    rospy.init_node('h_drive')
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
    margin = 0 # 100 # x
    points = []
    points.append([margin, ASSIST_BASE_LINE - ASSIST_BASE_WIDTH])
    points.append([margin, ASSIST_BASE_LINE + ASSIST_BASE_WIDTH])
    points.append([IMG_WIDTH-margin, ASSIST_BASE_LINE + ASSIST_BASE_WIDTH])
    points.append([IMG_WIDTH-margin, ASSIST_BASE_LINE - ASSIST_BASE_WIDTH])

    laneDetector = LaneDetector()
 
    #=========================================
    # 메인 루프 
    # 카메라 토픽이 도착하는 주기에 맞춰 한번씩 루프를 돌면서 
    # "이미지처리 +차선위치찾기 +조향각결정 +모터토픽발행" 
    # 작업을 반복적으로 수행함.
    #=========================================
    while not rospy.is_shutdown():

        # 이미지처리를 위해 카메라 원본이미지를 img에 복사 저장
        while img_ready == False:
            continue

        img = image.copy()  
        
        # 디버깅을 위해 모니터에 이미지를 디스플레이
        cv2.imshow("CAM View", img)
        cv2.waitKey(1)       
        img_ready = False

        angle, lane = laneDetector(img, 'left')
        print("angle", angle)
        print("lane", lane)
        #=========================================
        # IMAGE PROCESSING 
        #=========================================
        # birdView= perspectiveWarp(img) # birdseye view 적용
        
        # # show detected lines
        # img_line = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)

        # # thresholding
        # img_thres = np.copy(birdView) # image 복사

        # #  BGR 제한 값 설정
        # blue_threshold = 200
        # green_threshold = 200
        # red_threshold = 200
        # bgr_threshold = [blue_threshold, green_threshold, red_threshold]

        # img_canny_edge = Canny_Edge_Detection(img_thres) # canny
        # img_roi_canny_edge = region_of_interest(img_canny_edge, points) # ROI
        # cv2.imshow("ROI Edge Window", img_roi_canny_edge)
        # cv2.imshow("birdView", birdView)

        #=========================================
        # hough Transform을 사용한 차선 검출
        #=========================================
        # linesP = cv2.HoughLinesP(img_roi_canny_edge, 1, math.pi/180, 30, None, 15, 10)

        # # 인식된 차선이 없는 경우 angle=0, speed=7
        # if linesP is None:
        #     #drive(0, 7)
        #     continue


        #=========================================
        # y = 300(ROI)일 때 차선과의 교차점 표시
        # 검출한 차선 그리기
        #=========================================
        # print(linesP)
        
        # gradient = [] # 기울기
        # intersect = [] # 절편
        # intersect_base = [] # x = b + a*y

        # for i in range(len(linesP)):

        #     L = linesP[i][0].tolist()

	    # #if L[3] - L[1] == 0:
	    #   #  continue

        #     # 횡단보도 등 가로선 제외
        #     if -30 < L[3] - L[1] < 30:
        #         continue
        #     else:
        #         line_num = len(intersect_base)

        #     # print(L)
            
        #     gradient.append(((L[2] - L[0])*1.0) / ((L[3] - L[1])*1.0)) # 기울기 x/y
        #     intersect.append(L[0] - gradient[line_num] * L[1]) # x절편
        #     intersect_base.append(intersect[line_num] + gradient[line_num] * ASSIST_BASE_LINE) # x = b + a*y
        #     cv2.circle(img_line, (int(intersect_base[line_num]), ASSIST_BASE_LINE), 5, (255, 0, 0), -1) # 선과 y = 300의 교차점을 원으로 표시
        #     cv2.line(img_line, (L[0], L[1]), (L[2], L[3]), (255, 255, 255), 4, cv2.LINE_AA) # 차선을 img_line에 표시

        #=========================================
        # FIND CENTER
        # 연결된 픽셀들을 그룹화하여 무게중심을 찾고 중심점을 계산
        #=========================================
        # if len(linesP) != 0:
        #     cnt, _, stats, centroids = cv2.connectedComponentsWithStats(img_line, 8, cv2.CV_32S)
        #     print("cnt : " + str(cnt)) # 객체 수 + 1
        #     c_x_sum = 0
        #     LINE_exist = [0, 0] # ROI의 중심을 기준으로 왼쪽과 오른쪽에 차선 존재 여부를 저장

        #     if cnt == 1:
        #         continue

        #     # 레이블링 결과에 사각형과 넘버 표시하고 중심점 구하기.
        #     for i in range(1, cnt): # 각각의 객체 정보에 들어가기 위한 반복문, 범위를 1부터 시작한 이유는 배경을 제외
        #         # 평균으로 중심점 구하기
        #         area = stats[i][cv2.CC_STAT_AREA] # 픽셀 수
        #         c_x = centroids[i][0] # x 무게중심
        #         c_y = centroids[i][1] # y 무게중심
        #         print("Centroid" ,area, c_x, c_y)
        #         c_x_sum = c_x_sum + c_x # 평균내기위해 다 더함.

        #         if margin <= c_x <= IMG_WIDTH/2: # 왼쪽 차선이 보임
        #             LINE_exist[0] = 1
        #         elif IMG_WIDTH/2 - 50 <= c_x <= IMG_WIDTH - margin: # 오른쪽 차선이 보임
        #             LINE_exist[1] = 1
            
        #     line_center = c_x_sum / (cnt - 1) # 평균
        #     print("Centroid Center : " + str(line_center))


        #=========================================
        # angle 결정
        #=========================================
        # line_center, angle = center_to_angle(line_center, gradient, LINE_exist) # center의 x좌표 -> angle
        # cv2.line(img_line, (IMG_WIDTH//2, IMG_HEIGHT), (int(line_center), 330), (255, 255, 255), 4, cv2.LINE_AA) # 차의 위치로부터 center점까지 선으로 연결하여 시각화.


        #=========================================
        # 차량의 속도 값인 speed값 정하기.
        # 직선 코스에서는 빠른 속도로 주행하고 
        # 회전구간에서는 느린 속도로 주행하도록 설정함.
        # #=========================================
        speed = 3
        print("speed : " + str(speed))
        # drive() 호출. drive()함수 안에서 모터 토픽이 발행됨.
        # drive(angle, speed)

#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함.
# start() 함수가 실질적인 메인 함수임. 
#=============================================
if __name__ == '__main__':
    start()

