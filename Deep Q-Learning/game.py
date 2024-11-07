from abc import ABC, abstractmethod
import numpy as np
import random
from typing import Literal
class Game(ABC):
    def __init__(self) -> None:
        pass
    @abstractmethod
    def get_init_state(self, size_board):
        pass
    @abstractmethod
    def get_next_actions(self, state):
        pass
    @abstractmethod
    def is_terminal_state(self, state):
        pass
    @abstractmethod
    def get_reward(self, next_state):
        pass
    @abstractmethod
    def get_next_state(self, state, action):
        pass

from scipy.signal import correlate2d
class TicTacToe(Game):
    def _get_kernel(self, state: np.ndarray):
        '''
        Create kernels of the appropriate size for the state, min is 3, max is 5
        The kernels are unique a row or a column or a diagonal is 1
        '''
        size_board, _ = state.shape
        # arrays containing kernels
        rows_1 = []
        cols_1 = []
        kernel_size = min(size_board, 5)
        for i in range(kernel_size):
            row_1 = np.zeros((kernel_size, kernel_size))
            col_1 = row_1.copy()

            row_1[i,:] = 1
            col_1[:,i] = 1
            
            rows_1.append(row_1)
            cols_1.append(col_1)

        first_diagonal = np.zeros((kernel_size, kernel_size))
        np.fill_diagonal(first_diagonal, 1)

        second_diagonal = np.fliplr(first_diagonal)

        return [*rows_1, *cols_1, first_diagonal, second_diagonal]
    def get_init_state(self, size_board: int):
        '''
        input: size_board
        output: np.zeros((size_board, size_board)) or -1 in random location
        '''
        init_board = np.zeros((size_board, size_board))

        if random.random() > 0.5:
            # -1 in random location
            action = np.random.choice(range(0,9))
            init_board = self._get_action(init_board, action, -1)

        return init_board
    
    def get_next_actions(self, state: np.ndarray):
        '''
        Flatten the input state and return actionable positions: 0 -> size_borad**2-1
        input:
            state: np.ndarray
        output:
            actions: np.ndarray[0 -> state.size()-1]
        '''
        if self.is_terminal_state(state):
            return None
        
        new_state = state.flatten()

        actions = np.where(new_state == 0)[0]

        return actions
    
    def is_terminal_state(self, state: np.ndarray):
        '''
        Translate each kernel, if which kernel makes the value in that area the largest -> terminal there
        '''
        kernels = self._get_kernel(state)
        size_board, _ = state.shape
        
        for kernel in kernels:
            conv = correlate2d(state, kernel, mode='valid')
            
            result = min(size_board, 5)
            if result in conv or -result in conv:
                return True
        if 0 not in state:
            return True
        
        return False
    
    def _get_action(self, state: np.ndarray, action: int, player: Literal[1,-1]):
        size_board = state.shape[0]

        new_state = state.flatten()

        new_state[action] = player

        new_state = new_state.reshape(size_board, -1)

        return new_state
    
    def get_next_state(self, state: np.ndarray, action: int):
        '''environment returns random state'''
        
        # Check the end status, if true, no action will be taken
        if self.is_terminal_state(state):
            return None

        new_state = self._get_action(state, action, player=1)
        # check the end state, if true, the environment will not issue random actions
        if self.is_terminal_state(new_state):
            return new_state

        next_actions_opponent = self.get_next_actions(new_state)

        if len(next_actions_opponent) > 0:
            random_action = np.random.choice(next_actions_opponent)

            new_state = self._get_action(new_state, random_action, -1)

        return new_state

    def get_reward(self, next_state):
        size_board = next_state.shape[0]

        kernels = self._get_kernel(next_state)

        if self.is_terminal_state(next_state):
            for kernel in kernels:
                conv = correlate2d(next_state, kernel, mode='valid')
                
                result = min(size_board, 5)
                if result in conv:
                    return 1
                if -result in conv:
                    return -1
                
            if 0 not in next_state:
                return 0
        else:
            return 0