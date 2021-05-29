import random
import matplotlib.pyplot as plt
import numpy as np


def learning_curve(data, x_index=0, y1_index=1, y2_index=None, title='',
                   x_name='', y_name='', y1_legend='', y2_legend=''):
    """
    根据数据统计绘制学习曲线
    :param data: 数据元组，每一个元素是一个列表，各列表长度一直（[],[],[])
    :param x_index: x轴使用的数据list在元组中的索引值
    :param y1_index: y轴使用的数据list在元组中的索引值
    :param y2_index: y轴使用的数据list在元组中的索引值
    :param title: 图标名称
    :param x_name: x轴名称
    :param y_name: y轴名称
    :param y1_legend: y1图例
    :param y2_legend: y2图例
    :return: None
    """
    fig, ax = plt.subplots()  # 返回一个图对象figure,和坐标对象
    x = data[x_index]
    y1 = data[y1_index]
    ax.plot(x, y1, label=y1_legend)
    if y2_index is not None:
        ax.plot(x, data[y2_index], label=y2_legend)
    ax.grid(True, linestyle='-.')  # 是否打开网格
    ax.tick_params(labelcolor='black', labelsize='medium', width=1)  # 参数设置
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.legend()
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    plt.show()


def str_key(*args):
    new_arg = []
    for arg in args:
        if type(arg) in [list, tuple]:
            new_arg += [str(i) for i in arg]
        else:
            if arg is None:
                pass
            else:
                new_arg.append(str(arg))
    return '_'.join(new_arg)


def set_dict(target_dic, value, *args):
    if target_dic is None:
        return
    target_dic[str_key(*args)] = value


def get_dic(target_dic, *args):
    if target_dic is None:
        return
    return target_dic.get(str_key(*args), 0)


def unifom_random_pi(A, s=None, Q=None, a=None):
    """
    均一随机策略下的某行为概率
    """
    n = len(A)
    if n == 0:
        return 0.0
    return 1.0/n


def sample(A):
    """
    从A中随机选一个动作
    """
    return random.choice(A)


def uniform_random_policy(A, s=None, Q=None):
    return sample(A)


def greedy_pi(A, s, Q, a):
    """
    根据贪婪策略，计算在行为空间A中，状态s下，a行为被贪婪选中的机率
    注意：考虑多个行为价值相等的情况
    """
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = get_dic(Q, s, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)
    n = len(a_max_q)
    if n == 0:
        return 0.0
    return 1.0/n if a in a_max_q else 0.0


def greedy_policy(A, s, Q, epsilon=None):
    """
    在给定状态下，从行为空间中选择一个行为a，使得Q(s,a)=max(Q(s, a)
    注意：考虑多个行为价值相等的情况
    """
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = greedy_pi(A, s, Q, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)
    return random.choice(a_max_q)


def epsilon_greedy_pi(A, s, Q, a, epsilon=0.1):
    m = len(A)
    if m ==0:
        return 0.0
    greedy_p = greedy_pi(A, s, Q, a)
    if greedy_p == 0:
        return epsilon/m
    # n = int(1.0/greedy_p)
    return (1-epsilon)*greedy_p + epsilon/m


# def epsilon_greedy_policy(A, s, Q, epsilon=0.05):
#     rand_value = random.random()
#     if rand_value < epsilon:
#         return sample(A)
#     else:
#         return epsilon_greedy_pi(A, s, Q)



def epsilon_greedy_policy(A, s, Q, epsilon, show_randon_num=False):
    pis = []
    m = len(A)
    for i in range(m):
        pis.append(epsilon_greedy_pi(A, s, Q, A[i], epsilon))
    rand_value = random.random()
    for i in range(m):
        if show_randon_num:
            print('随机数:{:.2f}，拟减去概率{}'.format(rand_value, pis[i]))
        rand_value -= pis[i]
        if rand_value < 0:
            return A[i]
