import torch
import torch.nn as nn
from model import *
from train_dqn import *
import matplotlib.pyplot as plt

model_DQN = torch.load('./model/model_TicTacToe3x3.pth')
def predict(size_board):
    model_DQN.eval()
    num_games = 1000
    num_wins = 0
    num_draws = 0
    num_losts = 0

    result_game_lost = None
    for i in range(1, 1 + num_games):
        running = True
        current_state = game.get_init_state(size_board)
        result = 'not win'

        while(running):
            # lấy nước đi tốt nhất của Agent
            phi = phi_preprocess(current_state)
            with torch.no_grad():
                Q_values = model_DQN(torch.Tensor(phi))
            best_action = greedy(current_state, Q_values)
            # Cập nhật current_state
            current_state = game.get_next_state(current_state, best_action)
            r = game.get_reward(current_state)

            if game.is_terminal_state(current_state):
                running = False
                if r == 1:
                    num_wins += 1
                    result = 'win'
                elif r == -1:
                    num_losts += 1
                    result = 'lost'
                    result_game_lost = current_state
                else:
                    num_draws += 1
                    result = 'draw'
        print(f'game {i}: {result}')
    print(f"win rate: {(num_wins/num_games)*100:.3f}%", 
          f"lost rate: {(num_losts/num_games)*100:.3f}%", 
          f"draw rate: {(num_draws/num_games)*100:.3f}%", '\n',
          result_game_lost)
def plot(scores, name):
    plt.plot(scores)
    plt.xlabel("Episodes")
    plt.ylabel(f"{name}")
    plt.title(f"{name} Learning Curve")
    plt.show()
def smooth_vector(vec, window_size):
    return np.array([np.mean(vec[i:i+window_size]) for i in range(len(vec) - window_size + 1)])
if __name__ == '__main__':
    size_board = 3
    predict(size_board)
    with open('./results/result.txt', 'r') as file:
        datas = file.readlines()
    plot(smooth_vector(eval(datas[0]), 50), 'Reward')
    plot(smooth_vector(eval(datas[1]), 50), 'Loss')
    plot(eval(datas[2]), 'Epsilon')

    