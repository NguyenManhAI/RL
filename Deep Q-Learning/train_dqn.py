import torch
import torch.nn as nn
import numpy as np
from config import *
from model import *
from game import *

# Khởi tạo replay memory D với dung lượng N (Số exp) (hyperparameter)
from collections import deque
import random

class ReplayMemory:
    def __init__(self, capacity):
        # Khởi tạo buffer vòng với kích thước tối đa là capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, is_terminal_state):
        # Lưu trữ trải nghiệm (experience) vào replay memory
        experience = (state, action, reward, next_state, is_terminal_state)
        self.memory.append(experience)

    def sample(self, batch_size):
        # Lấy một minibatch ngẫu nhiên từ replay memory
        batch_size = min(batch_size, self.__len__())
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # Trả về số lượng trải nghiệm hiện có trong replay memory
        return len(self.memory)


# Khởi tạo action value function Q với random weight \theta
model_DQN = ResMLP(config_ResMLP_for_DQN).to(device) #\theta là tham số mô hình
game = TicTacToe()

def phi_preprocess(state: np.ndarray):
    return state.flatten()

def mask_action_output_DQN(actions: list, len_action_space: int):
    '''
    Implement a one-hot mask for the output, selecting only the output of the selected action
    '''
    mask = torch.zeros((len(actions),len_action_space))
    for i, action in enumerate(actions):
        mask[i, action] = 1
    return mask

import random
import math
from typing import Literal
def epsilon_greedy(state: np.ndarray, epsilon: float, Q_values: torch.Tensor):
    actions = game.get_next_actions(state)

    if random.random() < epsilon:
        # Chọn ngẫu nhiên hành động từ không gian hành động hợp lệ
        action = random.choice(actions)
    else:
        # Chọn hành động có giá trị Q cao nhất từ các hành động hợp lệ
        valid_Q_values = Q_values[actions]
        action = actions[torch.argmax(valid_Q_values).item()]
    
    return action

def greedy(state: np.ndarray, Q_values: torch.Tensor):
    actions = game.get_next_actions(state)
    valid_Q_values = Q_values[actions]
    action = actions[torch.argmax(valid_Q_values).item()]
    return action

def decay_epsilon(episode, M, mode = Literal['curve', 'linear', 'threshold'], rate_curve = 10, curved_direction = Literal['left', 'right', None]):
    b = -1 if curved_direction == 'left' else 1
    x = episode / M

    if mode == 'linear':
        m = 0.01
    if mode == 'curve':
        m = rate_curve
    if mode == 'threshold':
        if x < 0.5:
            return 0.9
        else:
            return 0.1
    term1 = 1 / (m * x - (0.9 * m + b * np.sqrt(0.81 * m**2 + 3.6 * m)) / 1.8)
    term2 = 1 + 1.8 / (0.9 * m + b * np.sqrt(0.81 * m**2 + 3.6 * m))
    epsilon = term1 + term2
    return epsilon

import torch.optim as optim
D = ReplayMemory(100)

def train(batch_size, M, C, size_board, gamma, config_decay_epsilon):
    # Khởi tạo replay memory D với dung lượng N (Số exp) (hyperparameter)
    # D = ReplayMemory(N)
    # Khởi tạo target action-value function \hat Q (fixed Q) với \theta^- = \theta
    Q_hat = ResMLP(config_ResMLP_for_DQN).to(device)
    Q_hat.load_state_dict(model_DQN.state_dict())

    # Khởi tạo train model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_DQN.parameters(), lr=0.001)

    # Khởi tạo epsilon
    epsilon_start = 1
    epsilon_min = 0.1
    epsilon = epsilon_start

    # Tắt gradient cho Q_hat
    for param in Q_hat.parameters():
        param.requires_grad = False

    step = 0
    model_DQN.train()
    ret = [] # list chứa tổng reward trong mỗi episode
    losses = []
    epsilons = []
    for episode in range(1, M+1): 
        s_t = game.get_init_state(size_board)
        # phi_t = phi_preprocess(s_t)
        running = True
        rewards_all = []
        losses_episode = []
        while(running):
            phi_t = phi_preprocess(s_t)
            # Sampling
            with torch.no_grad():
                Q_values = model_DQN(torch.Tensor(phi_t).to(device))
            # using \epsilon greedy to get action
            a_t = epsilon_greedy(s_t, epsilon, Q_values)
            # Execute action at in emulator and observe reward rt and state st+1
            s_t1 = game.get_next_state(s_t, a_t)
            r_t = game.get_reward(s_t1)
            rewards_all.append(r_t)
            is_terminal_state = False
            if game.is_terminal_state(s_t1):
                running = False
                is_terminal_state = True
            # phi_t1 = phi_preprocess(s_t1)
            # Store transition in D
            D.push(s_t.copy(), a_t, r_t, s_t1.copy(), is_terminal_state)
            # Cập nhật s_t:
            s_t = s_t1.copy()
            # Training
            # Compute y_j (Q_target)
            samples = D.sample(batch_size)
            Q_targets = []
            for i, sample in enumerate(samples):
                s_j, a_j, r_j, s_j1, is_terminal_state_j1 = sample
                #Q_hat(phi_j1), compute Q_target
                if not is_terminal_state_j1:
                    phi_j1 = phi_preprocess(torch.Tensor(s_j1)).to(device)
                    Q_values_j1 = Q_hat(phi_j1)
                    a_j1 = greedy(s_j1, Q_values_j1)
                    y_j = r_j + gamma*Q_values_j1[a_j1]
                    Q_targets.append(y_j)
                else:
                    Q_targets.append(r_j)
            Q_targets = torch.Tensor(Q_targets).to(device)
            # Compute Q value estimate:
            states, actions = list(zip(*samples))[:2]
            phi = torch.Tensor(np.array([phi_preprocess(state) for state in states])).to(device)
            Q_estimates = model_DQN(phi).to(device)
            mask = mask_action_output_DQN(actions, Q_estimates.shape[-1]).to(device)
            Q_estimates *= mask
            Q_estimates = Q_estimates.sum(-1)
            # Compute Q Loss
            optimizer.zero_grad()

            loss = criterion(Q_estimates, Q_targets)
            losses_episode.append(loss.item())

            loss.backward()  # Backpropagation
            optimizer.step()  # Update trọng số

            #reset Q_hat after C step:
            # Update step:
            step += 1
            if step % C == 0:
                Q_hat.load_state_dict(model_DQN.state_dict())

        epsilon = decay_epsilon(episode, M, **config_decay_epsilon)

        ret.append(np.sum(rewards_all))
        losses.append(np.average(losses_episode))
        epsilons.append(epsilon)
        if episode % 100 == 0:
            print(f"Episode {episode}: Avg Rewards:", np.mean(ret[-20:]), "Avg Loss:", np.mean(losses[-20:]), f"epsilon: {epsilon}")
    return ret, losses, epsilons

if __name__ == "__main__":
    print('training...')
    print(f'cuda: {torch.cuda.is_available()}')
    ret, losses, epsilons = train(config_decay_epsilon=config_decay_epsilon, **config_train_DQN_for_TicTacToe)
    with open('./results/result.txt', 'w') as file:
        file.write(str(ret)+ '\n' + str(losses) + '\n' + str(epsilons))
    torch.save(model_DQN, './model/model_TicTacToe3x3.pth')
    print('completed!')