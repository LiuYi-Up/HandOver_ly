from random import random, choice
from core import Agent
from gym import Env
from utils import str_key, set_dict, get_dic
from utils import epsilon_greedy_pi, epsilon_greedy_policy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pynput.keyboard import Listener
import keyboard

BATCH_SIZE = 4
LR = 0.01                   # learning rate
EPSILON = 1e-2              # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
N_ACTIONS = 2
N_STATES = 21*3+3
MEMORY_CAPACITY = 2000


class NetApproximator(nn.Module):  # approximator 近似者
    def __init__(self):
        """
        近似价值函数
        """
        super(NetApproximator, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=4)
        self.pool1d = nn.MaxPool1d(kernel_size=17 - 4 + 1)
        self.fc = nn.Linear(in_features=20, out_features=N_ACTIONS)
        # self.bn1 = nn.BatchNorm1d(self.inplanes)

        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        向前运算，根据网络的输入得到输出
        :param x: 输入特征
        """
        hand_points, dis = self._prepare_data(x)
        conv1 = self.conv1(hand_points)
        conv2 = self.conv2(conv1)
        pool1d_value = self.pool1d(conv2)
        pool = pool1d_value.view(-1, pool1d_value.size(1))
        

        relu = self.relu(self.fc2(self.relu(self.fc1(dis))))
        fc_input = torch.cat([pool, relu], axis=1)

        fc_output = self.fc(fc_input)
        return fc_output

    def _prepare_data(self, x):
        """
        将每个状态中的关键点和距离分开
        """

        if isinstance(x, np.ndarray):
            x = np.array(x).reshape((-1, 66))
    
        hand_points = x[:, :63]
        dis = x[:, -3:]

        hand_points = np.array(hand_points).reshape((-1, 21, 3))# 将1*63的数据转换为3*21的数据
        hand_points = torch.FloatTensor(hand_points).permute(0, 2, 1)
        dis = torch.FloatTensor(dis)

        return hand_points, dis

    def fit(self, x, y, criterion=None, optimizer=None, epochs=1, learning_rate=LR):
        """
        通过训练网络参数来拟合给定的输入x和输出y
        """
        if criterion is None:
            criterion = torch.nn.MSELoss(size_average=False)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if epochs < 1:
            epochs = 1
        # y = self._prepare_data(y)
        y = torch.Tensor(y)
        # print('------------ybatcn', y_batch)
        
        for t in range(epochs):
            y_pred = self.forward(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()  # 梯度重置，准备接收新梯度值
            loss.backward()  # 反向传播时自动计算相应节点的梯度
            optimizer.step()  # 更新权重
        return loss

    def __call__(self, x):
        y_pred = self.forward(x)
        return y_pred.data.numpy()

    def clone(self):
        """
        :return: 返回当前模型的深度拷贝对象
        """
        return copy.deepcopy(self)


class DQNAgent(Agent):
    """
    使用近似价值函数实现的Q-learning个体
    """
    def __init__(self, env: Env = None,
                 capacity=20000,
                 epochs=2):
        if env is None:
            raise str('agent should have an environment.')
        super(DQNAgent, self).__init__(env, capacity)

        # 行为网络，用来计算产生行为及对应的Q值，参数更新频繁:策略产生实际交互行
        # 为的依据
        self.behavior_Q = NetApproximator()
        if self.trained_model != '':
            self.behavior_Q.load_state_dict(torch.load(self.trained_model))
            self.losses = np.load(self.train_losses).tolist()
            print('model loded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', self.trained_model)
        # 计算目标价值的网络，初始时从行为网络拷贝而来， 两者参数一致， 该网络参数不定期更新
        # 该网络根据状态和行为得到目标价值，是计算代价的依据
        self.target_Q = self.behavior_Q.clone()
        self.batch_size = BATCH_SIZE
        self.epochs = epochs

    def _updata_target_Q(self):
        """
        将更新的策略Q网络（连带参数）复制给输出目标值Q的网络
        由于每个状态行为对的Q值是未知的，所以在学习过程中利用过去反复实验得到的结果Q作为目标值（相当于label）
        """
        self.target_Q = self.behavior_Q.clone()

    def policy(self, A, s, Q=None, epsilon=None):
        """
        根据更新策略的价值函数网络产生一个行为
        DQN生成行为的策略仍然是ε-greedy策略
        """
        Q_s = self.behavior_Q(s)
        print('---------Q_s-------------', Q_s)
        action = int(np.argmax(Q_s))
        rand_value = random()
        if epsilon is not None and rand_value < epsilon:  # 本例中使用数字0-4表示上下左右停止5个不同的动作
            return self.env.action_space.sample()
        else:
            Q_s = self.behavior_Q(s)
            action = int(np.argmax(Q_s))
        return action  # argmax返回最大值所对应的索引

    def learning_method(self, gamma=0.9, alpha=0.1, epsilon=1e-5, display=False, lambda_=None):
        print('环境初始化')
        self.state = self.env.reset()
        s0 = self.state

        time_in_episode, total_reward = 0, 0
        is_done = False
        loss = 0
        while not is_done:
            s0 = self.state  # 获取当前状态

            a0 = self.perform_policy(s0, epsilon)  # 基于 行为策略 产生行为
            s1, r1, is_done, info, total_reward = self.act(a0)  # 与环境交互

            if s1 is None:   # 当检测不到手时不计入状态中
                print('--------------------No hands------------------------------')
                continue

            if display:
                self.env.render()
#########################################1
            if self.total_trans > self.batch_size:
            # if self.total_trans > 1:
                # 总的转换数大于batch_size时，丢入网络计算loss，此后每进行一次转换都会进入网络学习
                # 因为此时才能随机选取batch_size个数据
                loss += self._learn_from_memory(gamma, alpha)  # 一个完整序列每一步转换的loss之和为总loss 
####################################################2
            time_in_episode += 1  # 一个序列里的转换数
            
        loss /= time_in_episode  # 平均每一转换的loss
        epoch_num = len(self.experience.episodes)
#####################################################1
        # if (epoch_num +1) % self.snapshot_epoch == 0:
        #         torch.save(self.behavior_Q.state_dict(), self.output_file+str(epoch_num+1150+1)+'.pkl')
        #         np.save(self.output_file+str(epoch_num+1150+1)+'-losses.npy', np.array(self.losses))
####################################################################2
        return time_in_episode, total_reward, loss

    def _learn_from_memory(self, gamma, learning_rate):
        trans_pieces = self.sample(self.batch_size)  # 随机即从某一序列中随机选取1个转换(list类型)，共操作batch_size次
        # 每个trans包含s0,a0,reward,is_done,s1

        states_0 = np.vstack([x.s0 for x in trans_pieces])  # 在竖直方向上堆叠
        actions_0 = np.array([x.a0 for x in trans_pieces])
        reward_1 = np.array([x.reward for x in trans_pieces])
        is_done = np.array([x.is_done for x in trans_pieces])
        states_1 = np.vstack([x.s1 for x in trans_pieces])

        # 准备训练
        x_batch = states_0
        # y_batch得到numpy格式的结果,target_Q网络由前一次历史behavior_Q网络克隆得到
        # 由历史结果估算得到一个Q
        y_batch = self.target_Q(states_0)
        print('----------------------y_batch--------------', y_batch)

        # is_done时Q_target==reward_1
        Q_target = reward_1 + gamma * np.max(self.target_Q(states_1), axis=1) * (~is_done)  # 这里是多维数组在计算
        print('=======Q_target==========', Q_target)
        print('------------reward_1---------', reward_1)

        # # 取消下面代码行柱视则变为DDQN,行为a'从行为价值网络中得到
        # a_prime = np.argmax(self.behavior_Q(states_1), axis=1).reshape(-1)
        # # (s',a')的价值从目标价值网络中得到
        # Q_states_1 = self.target_Q(states_1)
        # temp_Q = Q_states_1[np.arange(len(Q_states_1)), a_prime]
        # # （s,a)的目标价值根据贝尔曼方程得到
        # Q_target = reward_1 + gamma *temp_Q * (~ is_done)

        y_batch[np.arange(len(x_batch)), actions_0] = Q_target  # 这一步将上一步计算得到的Q更新到上上步的Q中，相当于只有一部分更新？
        
        # 训练  行为价值网络  ，更新参数
        loss = self.behavior_Q.fit(x=x_batch,
                                   y=y_batch,
                                   learning_rate=learning_rate,
                                   epochs=self.epochs)

        mean_loss = loss.sum().item() / self.batch_size

        self._updata_target_Q()

        return mean_loss