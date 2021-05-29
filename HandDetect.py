import mediapipe as mp
import cv2
import torch
import math
import pyrealsense2 as rs
import numpy as np
import os
import time


class RealsenseCamera():

    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)
        self.center = [0, 0, 0]
        self.xc = 0
        self.yc = 0
        self.zc = 0
        self.yaw_angle = 0
        self.pitch_angle = 0

    def pixel2point(self, color_frame, depth_frame, u, v):
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        point = [u, v]

        depth = depth_frame.get_distance(int(point[0]/2), int(point[1]/2))
        point = np.append(point, depth)

        if depth != 0:
            x = point[0]
            y = point[1]
            z = point[2]
            x, y, z = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], z)
            self.center = [x, y, z]
        return self.center

    def get_images(self):
        align_to = rs.stream.color
        align = rs.align(align_to)

        frames = self.pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()

        return self.color_frame, self.depth_frame


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


class HandLandMarkDetect():

    def __init__(self):
        self.camera = RealsenseCamera()
        self.hand_point = None
        self.hand_pixel = None
        self.hand_landmarks = None
        self.color_image = None

    def get_HandLandMarks(self):

        color_frame, depth_frame = self.camera.get_images()
        self.color_image = np.asanyarray(color_frame.get_data())
        self.color_image = cv2.cvtColor(cv2.flip(self.color_image, 1), cv2.COLOR_BGR2RGB)
        image_width, image_height = self.color_image.shape[0], self.color_image.shape[1]
    
        results = hands.process(self.color_image)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR)
        # print('0=0=0=0=0', self.color_image)
        hand_point, hand_pixel = None, None
        if results.multi_hand_landmarks:
            self.hand_landmarks = results.multi_hand_landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                hand_point = {
                    0: [],
                    1: [],
                    2: [],
                    3: [],
                    4: [],
                    5: []
                }

                hand_pixel = {
                    0: [],
                    1: [],
                    2: [],
                    3: [],
                    4: [],
                    5: []
                }

                hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
                hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
                hand_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

                hand_pixel[0].append([hand_x, hand_y, hand_z])

                hand_point_xyz = self.camera.pixel2point(color_frame, depth_frame, hand_x, hand_y)
                hand_point[0].append(hand_point_xyz)

                count = 0
                landmark_idx = 1

                for index in range(1, 21):
                    hand_x = hand_landmarks.landmark[index].x * image_width
                    hand_y = hand_landmarks.landmark[index].y * image_height
                    hand_z = hand_landmarks.landmark[index].z

                    hand_pixel[landmark_idx].append([hand_x, hand_y, hand_z])

                    hand_point_xyz = self.camera.pixel2point(color_frame,depth_frame, hand_x, hand_y)
                    hand_point[landmark_idx].append(hand_point_xyz)

                    count = count + 1
                    if count == 4:
                        count = 0
                        landmark_idx = landmark_idx + 1
        self.hand_point, self.hand_pixel = hand_point, hand_pixel
        return self.hand_point, self.hand_pixel

    def show_hand(self):
        if self.hand_landmarks:
            for hand_landmark in self.hand_landmarks:
                mp_drawing.draw_landmarks(self.color_image, hand_landmark, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('robotCamera', self.color_image)
        cv2.waitKey(5)

