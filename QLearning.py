import numpy as np
import pandas as pd
import math
import scipy.io as scio
import matplotlib.pyplot as plt

np.random.seed(0)

BERLimit = 0.001  # QoS
# 环境参数
A_SNR = 12  # 12sinx+9
B_SNR = 9
N_STATES = 20  # 状态：传输次数
ACTIONS = range(3)  # 动作：三种调制方式

ALPHA = 0.1     # 学习率
GAMMA = 0.9     # 之前奖励的衰减值
MAX_EPISODES = 1000   # 回合数


def build_q_table():
    table = pd.DataFrame(    # 使用pandas创建一个表格
        np.zeros((N_STATES, len(ACTIONS))),     # 全0初始化Q表，行为状态个数，列为动作个数
        columns=ACTIONS,    # index 行索引 columns 列索引
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table, epsilon):    # 根据state选择动作actions
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]  # 按索引号从q表中取所有动作，state：目前位置
    if (np.random.uniform() < epsilon) or ((state_actions == 0).all()):  # 随机数大于0.9即10%几率或state对应值全为0使用随机动作
        action_name = np.random.choice(ACTIONS)    # 随机选择动作
    else:   # 90%几率
        action_name = state_actions.idxmax()    # 使用Q表中state对应值最大的动作
    return action_name


def update_env(step):  # 环境变化
    snr = A_SNR * math.sin(step/10) + B_SNR
    return snr


def update_epsilon(epsilon):  # 探索率逐渐下降
    if epsilon > 0.001:
        epsilon *= 0.99
    return epsilon


def get_env_feedback(S, A, Data):    # 获得下一状态、奖励、已传输的数据
    snr = update_env(S)
    if int(A / 2):
        BER = math.erfc(math.sqrt(6*snr)*math.sin(math.pi/16))
    elif A:
        BER = 1-(1-0.5 * math.erfc(math.sqrt(snr)))**2
    else:
        BER = 0.5 * math.erfc(math.sqrt(snr))

    if BER <= BERLimit:
        R = -1/(A+1)  # 这里使用负的奖励值可以使agent倾向于使用未尝试过的动作，比R=A+1收敛速度快约10倍
        Data += A+1
    else:
        R = -5
    return S + 1, R, Data


def rl():
    # main part of RL loop
    qTable = build_q_table()    # 创建Q表
    result = np.zeros((MAX_EPISODES, 2))    # 该回合传输用的次数
    epsilon = 1  # 贪婪度，0.1几率随机选择动作
    for episode in range(MAX_EPISODES):     # MAX_EPISODES个回合的循环
        S = 0  # 初始状态
        data = 0
        is_terminated = False
        epsilon = update_epsilon(epsilon)  # 更新探索率
        while not is_terminated:
            A = choose_action(S, qTable, epsilon)  # 选择动作
            S_, R, data = get_env_feedback(S, A, data)  # 获得之后的状态及奖励

            if S_ != N_STATES:
                q_target = R + GAMMA * qTable.iloc[S_, :].max()  # q-leaning 真实值
            else:
                is_terminated = True  # 传输结束
                q_target = R
                times = "回合：" + str(episode) + "  传输数据量：" + str(data)
                result[episode, :] = (episode+1, data)  # 该episode回合的传输数据量
                print('\r{}'.format(times), end='')
            q_predict = qTable.loc[S, A]  # 估计值
            qTable.loc[S, A] += ALPHA * (q_target - q_predict)  # 更新Q表
            S = S_  # 更新状态
    return qTable, result


if __name__ == "__main__":
    Q_TABLE, Result = rl()
    # 保存数据
    dataNew = '/Q_Table.csv'
    resultNew = '/result.mat'
    scio.savemat(resultNew, {'Result': Result})
    Q_TABLE.to_csv(dataNew)
    # 画图
    plt.plot(Result[:, 0], Result[:, 1], color='r', linewidth=1.5)
    plt.xlabel('episode')
    plt.ylabel('data')
    plt.grid(1)
    plt.show()
