import numpy as np
import pandas as pd
import math
import os
import scipy.io as scio

np.random.seed(0)
fileDir = os.getcwd()
ACTIONS = ['H-BPSK', 'L-BPSK', 'H-QPSK', 'L-QPSK', 'H-8PSK', 'L-8PSK']  # 三种调制方式
stateData = np.array([3, 5, 7, 11, 13, 15, 17, 23, 25, 27])
ENERGY = 400
BERLimit = 0.001
N_STATES = ENERGY  # 状态数量
EPSILON = 0.9   # 贪婪度，90%几率选择最优动作
ALPHA = 0.1     # 学习率
GAMMA = 0.9     # 之前奖励的衰减值
MAX_EPISODES = 5000   # 回合数


def build_q_table(n_states, actions):
    table = pd.DataFrame(    # 使用pandas创建一个表格
        np.zeros((n_states, len(actions))),     # 全0初始化Q表，行为状态个数，列为动作个数
        columns=actions,    # index 行索引 columns 列索引
    )
    # print(table)    # show table
    return table


# def choose_action(state, q_table, counter):    # 根据state选择动作actions
#     # This is how to choose an action
#     state_actions = q_table.iloc[state, :]  # 按索引号从q表中取所有动作，state：目前位置
#     if counter > 3800:
#         epsl = 1
#     else:
#         epsl = EPSILON
#
#     if (np.random.uniform() > epsl) or ((state_actions == 0).all()):  # 随机数大于0.9即10%几率或state对应值全为0使用随机动作
#         action_name = np.random.choice(ACTIONS)    # 随机选择动作
#     else:   # 90%几率
#         action_name = state_actions.idxmax()    # 使用Q表中state对应值最大的动作
#     return action_name


def choose_action(state, q_table):    # 根据state选择动作actions
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]  # 按索引号从q表中取所有动作，state：目前位置
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # 随机数大于0.9即10%几率或state对应值全为0使用随机动作
        action_name = np.random.choice(ACTIONS)    # 随机选择动作
    else:   # 90%几率
        action_name = state_actions.idxmax()    # 使用Q表中state对应值最大的动作
    return action_name


def choose_snr(stepCounter):
    count = stepCounter % 20
    rand = np.random.uniform()
    if count < 10:
        snrNum = stateData[count]
    else:
        count = 19 - count
        snrNum = stateData[count]
    if rand < 0.2:
        rand_snr = np.random.randint(-1, 2)
    elif rand < 0.5:
        rand_snr = np.random.randint(-2, 3)
    elif rand < 0.8:
        rand_snr = np.random.randint(-3, 4)
    else:
        rand_snr = 0
    snr = snrNum + rand_snr
    return snr


def get_env_feedback(S, A, energy, Data):    # 动作对状态环境的影响
    snr = choose_snr(S)
    if A == 'H-BPSK':
        energy_snr = snr + 2
        BER = 0.5 * math.erfc(math.sqrt(energy_snr))
        energy = energy - 2
        if BER <= BERLimit:
            R = (1-BER) * 2 * math.log(2, 2)
            Data += 2
        else:
            R = -10
    elif A == 'L-BPSK':
        energy_snr = snr + 1
        BER = 0.5 * math.erfc(math.sqrt(energy_snr))
        energy -= 1
        if BER <= BERLimit:
            R = (1-BER) * 1 * math.log(2, 2)
            Data += 1
        else:
            R = -10
    elif A == 'H-QPSK':
        energy_snr = snr + 2
        BER = math.erfc(math.sqrt(4*energy_snr)*math.sin(math.pi/8))
        energy = energy - 2
        if BER <= BERLimit:
            R = (1-BER) * 4 * math.log(4, 2)
            Data += 4
        else:
            R = -10
    elif A == 'L-QPSK':
        energy_snr = snr + 1
        BER = math.erfc(math.sqrt(4*energy_snr)*math.sin(math.pi/8))
        energy -= 1
        if BER <= BERLimit:
            R = (1-BER) * 3 * math.log(4, 2)
            Data += 3
        else:
            R = -10
    elif A == 'H-8PSK':
        energy_snr = snr + 2
        BER = math.erfc(math.sqrt(6*energy_snr)*math.sin(math.pi/16))
        energy = energy - 2
        if BER <= BERLimit:
            R = (1-BER) * 6 * math.log(8, 2)
            Data += 6
        else:
            R = -10
    else:
        energy_snr = snr + 1
        BER = math.erfc(math.sqrt(6*energy_snr)*math.sin(math.pi/16))
        energy -= 1
        if BER <= BERLimit:
            R = (1-BER) * 5 * math.log(8, 2)
            Data += 5
        else:
            R = -10
    S_ = ENERGY - energy     # 更新状态
    return S_, R, energy, Data


def rl():
    # main part of RL loop
    qTable = build_q_table(N_STATES, ACTIONS)    # 创建Q表
    qTable_temp = qTable.copy()
    result = np.zeros((MAX_EPISODES, 2))    # 该回合传输用的次数
    for episode in range(MAX_EPISODES):     # MAX_EPISODES个回合的循环
        energy = ENERGY
        step_counter = 0
        S = 0  # 初始位置
        data = 0
        is_terminated = False
        while not is_terminated:
            A = choose_action(S, qTable)  # 选择动作
            S_, R, energy, data = get_env_feedback(S, A, energy, data)  # 获得之后的状态及奖励
            if energy <= 0:  # 能量用光或数据传完则结束这一回合
                S_ = 'terminal'  # 添加传输完成标志
            q_predict = qTable.loc[S, A]  # 估计值
            if S_ != 'terminal':
                q_target = R + GAMMA * qTable.iloc[S_, :].max()  # q-leaning 真实值
            else:
                is_terminated = True  # 传输结束
                q_target = R
                cishu = "次数：" + str(step_counter) + "  回合：" + str(episode) + "  传输数据量：" + str(data)
                result[episode, :] = (episode+1, data)  # 该episode回合的传输数据量
                print('\r{}'.format(cishu), end='')
            qTable.loc[S, A] += ALPHA * (q_target - q_predict)  # 更新Q表
            S = S_  # 更新状态
            step_counter += 1
        if episode == (MAX_EPISODES-2):
            qTable_temp = qTable.copy()
    error_calculate(qTable_temp, qTable)
    return qTable, result


def error_calculate(table_first, table_last):
    table_first_np = table_first.values
    table_last_np = table_last.values
    print("  MAE：" + str(np.mean(np.abs(table_last_np - table_first_np))))


if __name__ == "__main__":
    Q_TABLE, Result = rl()
    dataNew = fileDir + '\\Q_Table.csv'
    resultNew = fileDir + '\\result.mat'
    scio.savemat(resultNew, {'Result': Result})
    Q_TABLE.to_csv(dataNew)
