from utils import learning_curve
from agents import DQNAgent
from RobotWorld import RobotEnv


env = RobotEnv()
agent = DQNAgent(env)
data = agent.learning(gamma=0.99,
                      epsilon=5e-2,
                      decaying_epsilon=False,
                      alpha=1e-3,
                      max_episode_num=2000,
                      display=True,
                      snapshot_epoch=50,
                      output_file='/home/lab/Python_pro/Ly_pro/HandOver_ly/checkpoints/',
                      trained =True,
                      trained_model='/home/lab/Python_pro/Ly_pro/HandOver_ly/checkpoints/100.pkl',
                      train_losses='/home/lab/Python_pro/Ly_pro/HandOver_ly/checkpoints/100-losses.npy')
# learning_curve(data, 2, 1, title='DQN', x_name='episodes', y_name='reward of episode')

