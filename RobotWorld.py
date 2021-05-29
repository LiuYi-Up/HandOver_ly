import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from X_arm import X_arm
from HandDetect import HandLandMarkDetect


class RobotEnv():

    def __init__(self):
        self.X_arm = X_arm()
        self.X_arm.Open_port('/dev/ttyUSB2')
        self.hand_detect = HandLandMarkDetect()

        self.reward = 0  
        self.action = None  
        self.state = None

        # 0,1represent open, stay
        self.action_space = spaces.Discrete(2)  # 创建一个2维离散空间
        self.gripper=[0.12, 0.06, 0.64]
        # self.gripper=[0,0,0]

        self.seed()  # 产生一个随机数种子
        self.reset()

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        # self.X_arm.setJointAngle(7, 30, 0.5)
        # self.X_arm.setJointAngle(3, 0, 0.5)

        # close = input("物品已放置(y/n): ")
        # if close == 'y':
        #     self.X_arm.setJointAngle(7, 76, 0.5)
        #     self.X_arm.setJointAngle(3, 15, 0.5)
        #     pass
        i =0
        while True:
            hand_point, hand_pixel = self.hand_detect.get_HandLandMarks()
            self.hand_detect.show_hand()
            i +=1
            if hand_point and hand_pixel :
                print('---------------------',i)
                i=0
                break
        hand_state = self._transition(hand_point, hand_pixel)

        self.state = hand_state
        print('----------------新一局epoch开始-------------------')
        return np.array(self.state)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        self.hand_detect.show_hand()
        self.action = action  
        self.reward = 2
        done = False
            
        # if action == 0:
        #     init_reward = 0
        #     print('选择动作-stay')

        # if action == 1:
        #     # self.X_arm.setJointAngle(7, 30, 0.5)
        #     done = True
        #     person_reward = input("输入此次动作-open的评分： ")
        #     # print('state', self.state)

        hand_point, hand_pixel = self.hand_detect.get_HandLandMarks()
        if hand_pixel is None or hand_point is None:
            hand_state = None
        else:
            hand_state = self._transition(hand_point, hand_pixel)
            self.state = hand_state  

        if action == 0 and hand_state:
            distance = self._compute_dis(hand_state[-3], hand_state[-2], hand_state[-1])
            if distance <= 0.30:
                self.reward = 2
            else:
                self.reward = abs(3.6 - distance * 6)

            print('选择动作-stay', self.reward)

        if action == 1:
            # self.X_arm.setJointAngle(7, 30, 0.5)
            # self.X_arm.setJointAngle(3, 0, 0.5)
            done = True
            #######################################1
            person_reward = input("输入此次动作-open的评分： ")
            if person_reward == '\x1b' or person_reward == '\x1b\x1b' or person_reward == '\x1b\x1b\x1b':
                self.reward == 3
            else:
                self.reward = int(person_reward)
            ##############################################2
            # self.reward = 0.7*int(person_reward) + 0.3*init_reward
            # print('state', self.state)
        # if hand_state:
        #     distance = self._compute_dis(hand_state[-3], hand_state[-2], hand_state[-1])
            # print('**********distance*****************', distance)

        

        return np.array(self.state), self.reward, done, {}

    def _transition(self, hand_point, hand_pixel):
        hand_state = []

        hand_state.append(list(np.squeeze(hand_point[0])))

        # 从下到上将手指包络添加
        for index in range(4):
            hand_state.append(hand_point[1][index])
            hand_state.append(hand_point[2][index])
            hand_state.append(hand_point[3][index])
            hand_state.append(hand_point[4][index])
            hand_state.append(hand_point[5][index])

        # 计算手到机械爪的距离
        hand_state = np.array(hand_state)
        handCenter = np.array(hand_state.sum(0))
        handCenter[0] = handCenter[0] / 21
        handCenter[1] = handCenter[1] / 21
        handCenter[2] = handCenter[2] / 21

        distance_x = abs(handCenter[0] - self.gripper[0])
        distance_y = abs(handCenter[1] - self.gripper[1])
        distance_z = abs(handCenter[2] - self.gripper[2])

        # distance_x = abs(handCenter[0] - self.gripper[0])
        # distance_y = abs(handCenter[1] - self.gripper[1])
        # distance_z = abs(handCenter[2] - self.gripper[2])

        # 将距离添加到状态向量
        hand_state = hand_state.reshape((1, 21 * 3)).tolist()[0]
        hand_state.append(distance_x)
        hand_state.append(distance_y)
        hand_state.append(distance_z)
        return hand_state

    def _random_pos(self):
        return self.np_random.uniform(low=0, high=self.l_unit)

    def _compute_dis(self, dx, dy, dz):
        return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2) + math.pow(dz, 2))

    def render(self):
        """
        渲染
        """
        self.hand_detect.show_hand()

