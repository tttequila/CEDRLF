import MELD
import argparse
import imp
import os
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl

#####################  hyper parameters  ####################

RANDOMSEED = 1              # random seed

LR_A = 0.001                # learning rate for actor
LR_C = 0.002                # learning rate for critic
GAMMA = 1                 # reward discount
TAU = 0.01                  # soft replacement
MEMORY_CAPACITY = 5000     # size of replay buffer
BATCH_SIZE = 32             # update batchsize

MAX_EPISODES = 20000         # total number of episodes for training
TEST_EPISODES = 1000
MAX_EP_STEPS = 100          # total number of steps for each episode
TEST_PER_EPISODES = 100      # test the model per episodes
VAR = 1                  # control exploration
WINDOW_SIZE = 1

DATA_PATH = 'E:/Code/datasets/MELD.Features.Models/features'
LABEL_EMB = 'NRC_VAD'

###############################  DDPG  ####################################

class DDPG(object):
    """
    DDPG class
    """
    def __init__(self, a_dim, s_dim, a_bound):
        # memory用于储存跑的数据的数组：
        # 保存个数MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：分别是两个state，一个action，和一个reward
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # 建立actor网络，输入s，输出a
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=512, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=256, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l2')(x)
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l3')(x)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x)            #注意这里，先用tanh把范围限定在[-1,1]之间，再进行映射
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

        #建立Critic网络，输入s，a。输出Q值
        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=128, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        #更新参数，只用于首次赋值，之后就没用了
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        #建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        #建立critic_target网络，并和actor参数一致，不能训练
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        #建立ema，滑动平均值
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)


    def ema_update(self):
        """
        滑动平均更新
        """
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        paras = self.actor.trainable_weights + self.critic.trainable_weights    #获取要更新的参数包括actor和critic的
        self.ema.apply(paras)                                                   #主要是建立影子参数
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))                                       # 用滑动平均赋值

    # 选择动作，把s带进入，输出a
    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        return self.actor(np.array([s], dtype=np.float32))[0]

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    #随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]                    #根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]                         #从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  #从bt获得数据a
        br = bt[:, -self.s_dim - 1:-self.s_dim]         #从bt获得数据r
        bs_ = bt[:, -self.s_dim:]                       #从bt获得数据s'
        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        print(bs.shape, ba.shape)
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor：
        # Actor的目标就是获取最多Q值的。
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = tf.reduce_mean(q)  # 【敲黑板】：注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.ema_update()


    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        # 整理s，s_,方便直接输入网络计算
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        #把s, a, [r], s_横向堆叠
        transition = np.hstack((s, a, [r], s_))

        #pointer是记录了曾经有多少数据进来。
        #index是记录当前最新进来的数据位置。
        #所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        #把transition，也就是s, a, [r], s_存进去。
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model_VAD'):
            os.makedirs('model_VAD')

        tl.files.save_weights_to_hdf5('model_VAD/ddpg_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model_VAD/ddpg_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5('model_VAD/ddpg_critic.hdf5', self.critic)
        tl.files.save_weights_to_hdf5('model_VAD/ddpg_critic_target.hdf5', self.critic_target)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model_VAD/ddpg_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model_VAD/ddpg_actor_target.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order('model_VAD/ddpg_critic.hdf5', self.critic)
        tl.files.load_hdf5_to_weights_in_order('model_VAD/ddpg_critic_target.hdf5', self.critic_target)


def find_label(dic, a):
    label_dict = dic
    distance = 9999
    true_label = "nul"
    for emo, value in label_dict.items():
        new_distance = np.linalg.norm(a-value)
        if new_distance < distance:
            true_label = emo
            distance = new_distance
    return true_label

if __name__ == '__main__':
    
    #初始化环境

    env = gym.make('MELD-v0',
               dataset_path=DATA_PATH,
               partition='train',
               label_emb=LABEL_EMB,
               predefined_feat=True)
    
    env = env.unwrapped
  # p = env.reset()
    # reproducible，设置随机种子，为了能够重现
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED+1)

    #定义状态空间，动作空间，动作幅度范围
    a_dim = 3
    s_dim = 1911*WINDOW_SIZE + a_dim
    # print(s_dim)
    a_bound = 1

    #用DDPG算法
    ddpg = DDPG(a_dim, s_dim, a_bound)

    #训练部分：

    reward_buffer = []      #用于记录每个EP的reward，统计变化
    t0 = time.time()        #统计时间
    for i in range(MAX_EPISODES):
        window = [np.zeros(1911)]*WINDOW_SIZE
        t1 = time.time()
        s, done, _, _, _, _ = env.reset()
        ep_reward = 0
        cont = 0       #记录当前EP的reward
        a = env.label_dict['neutral']
        s = np.hstack(s)
        
        # shifting window
        window.pop(0)
        window.append(s)

        while not done:
                # Add exploration noise
            
            window_s = np.hstack((a, np.hstack(window)))
            # print(window_s.shape)
            a = ddpg.choose_action(window_s)       #这里很简单，直接用actor估算出a动作

                # 为了能保持开发，这里用了另外一种方式增加探索。
                # 因此需要需要以a为均值，VAR为标准差，建立正态分布，再从正态分布采样出a
                # 因为a是均值，所以a的概率是最大的。但a相对其他概率由多大，是靠VAR调整。这里我们其实可以增加更新VAR，动态调整a的确定性
                # 然后进行裁剪
            a = np.clip(np.random.normal(a, VAR), -1, 1)  
                # 与环境进行互动
            s_, done, _, _, _, r = env.step(a)
            r = -r
            if done:
                print(
                    '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f} | True label: {}    '.format(
                        i, MAX_EPISODES, ep_reward,
                        time.time() - t1, find_label(env.label_dict, a)
                    ), end = ""
                )
                continue
            s_ = np.hstack(s_)
            window.pop(0)
            window.append(s_)
            window_s_ = np.hstack((a, np.hstack(window)))
                # 保存s，a，r，s_
            ddpg.store_transition(window_s, a, r/5, window_s_)
            if not ddpg.pointer % MEMORY_CAPACITY:
                ddpg.learn()
                # 第一次数据满了，就可以开始学习
            

                #输出数据记录
            s = s_  
            ep_reward += r  #记录当前EP的总reward
            cont += 1
            
            
            # test
        if i and not i % TEST_PER_EPISODES:

            t1 = time.time()
            s,done, _, _, _, _ = env.reset()
            
            ep_reward = 0
            cont = 0
            emo_lst = []
            while not done:
                s = np.hstack(s)
                window.pop(0)
                window.append(s)
                
                window_s = np.hstack((a, np.hstack(window)))
                a = ddpg.choose_action(window_s)       #这里很简单，直接用actor估算出a动作
                a = np.random.normal(a, 0)
                pred_emo = find_label(env.label_dict, a)
                s_, done, _, true_emo, _, r = env.step(a)
                
                emo_lst.append(pred_emo)
                
                r = -r
                s_ = np.hstack(s_)
                    
                s = s_
                ep_reward += r
                cont += 1
                if done:
                    print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f} | True label: {}     '.format(
                            i, MAX_EPISODES, ep_reward/cont,
                            time.time() - t1, emo_lst
                        )
                    )

                    reward_buffer.append(ep_reward/cont)    

    if reward_buffer:
        plt.ion()
        plt.cla()
        plt.title('DDPG')
        plt.plot(np.array(range(len(reward_buffer))) * TEST_PER_EPISODES, reward_buffer)  # plot the episode vt
        plt.xlabel('episode steps')
        plt.ylabel('normalized state-action value')
        plt.ylim(-2,1)
        # plt.show()
        # plt.pause(0.01)
        plt.ioff()
        plt.show()
    print('\nRunning time: ', time.time() - t0)
    ddpg.save_ckpt()

    # test
    env = gym.make('MELD-v0',
               dataset_path=DATA_PATH,
               partition='test',
               label_emb=LABEL_EMB,
               predefined_feat=True)
    env = env.unwrapped
    total = {'anger':0, 'joy':0, 'sadness':0, 'surprise':0, 'fear':0, 'disgust':0, 'neutral':0}
    corr =  {'anger':0, 'joy':0, 'sadness':0, 'surprise':0, 'fear':0, 'disgust':0, 'neutral':0}
    emo_lst = []

    for i in range(TEST_EPISODES):
        window = [np.zeros(1911)]*WINDOW_SIZE
        s, done, _, _, _, _ = env.reset()
        ep_reward = 0
        cont = 0       #记录当前EP的reward
        a = env.label_dict['neutral']
        s = np.hstack(s)
        
        # shifting window
        window.pop(0)
        window.append(s)

        while not done:
                # Add exploration noise
            
            window_s = np.hstack((a, np.hstack(window)))
            # print(window_s.shape)
            a = ddpg.choose_action(window_s)       #这里很简单，直接用actor估算出a动作

                # 为了能保持开发，这里用了另外一种方式增加探索。
                # 因此需要需要以a为均值，VAR为标准差，建立正态分布，再从正态分布采样出a
                # 因为a是均值，所以a的概率是最大的。但a相对其他概率由多大，是靠VAR调整。这里我们其实可以增加更新VAR，动态调整a的确定性
                # 然后进行裁剪
            a = np.clip(np.random.normal(a, VAR), -1, 1)  
                # 与环境进行互动
            pred_emo = find_label(env.label_dict, a)
            s_, done, _, true_emo, _, r = env.step(a)
            
            emo_lst.append(pred_emo)
            total[true_emo] += 1
            if true_emo == pred_emo: corr[true_emo] += 1
            
            r = -r
            if done:
                print(
                    '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | True label: {}    '.format(
                        i, TEST_EPISODES, ep_reward, find_label(env.label_dict, a)
                    ), end = ""
                )
                continue
            s_ = np.hstack(s_)
            window.pop(0)
            window.append(s_)
            window_s_ = np.hstack((a, np.hstack(window)))
                # 保存s，a，r，s_
            s = s_  
            ep_reward += r  #记录当前EP的总reward
            cont += 1
            
            # test    
    
    with open("correct rate_VAD.txt", "w") as f:
        f.write(str(total))
        f.write('\n')
        f.write(str(corr))
