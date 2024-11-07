from typing import Literal
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from game import *
import numpy as np

app = Flask(__name__)
class Player(ABC):
    def __init__(self) -> None:
        pass
    @abstractmethod
    def choose_action(self, state):
        pass

class RandomPlayerTicTacToe(Player):
    def __init__(self, name: Literal['X', 'O']) -> None:
        self.name = name
        super().__init__()
    def choose_action(self, state: np.ndarray):
        actions = TicTacToe().get_next_actions(state)

        new_state = state.flatten()
        action = np.random.choice(actions)
        new_state[action] = 1 if self.name == 'X' else -1
        new_state = new_state.reshape(state.shape[0], -1)
        return new_state
    
class HumanPlayerTicTacToe(Player):
    def __init__(self, name: Literal['X', 'O']) -> None:
        super().__init__()
        self.name = name
    
    @app.route('/human-move', methods=['GET','POST'])
    def choose_action(self, state: np.ndarray):
        if request.method == 'POST':

            size_board = state.shape
            actions = TicTacToe().get_next_actions(state)

            row = int(request.form.get("row"))
            col = int(request.form.get("col"))

            action = (row + 1)*size_board + col
            if action not in actions:
                flash("Invalid location!", "message")
                return None

            new_state[action] = 1 if self.name == 'X' else -1
            new_state = new_state.reshape(state.shape[0], -1)
            return new_state
        
        return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get data from form
        player_symbol = request.form.get('player_symbol')
        board_size = int(request.form.get('board_size'))

        # Go to the game page with parameters
        return redirect(url_for('game', player_symbol=player_symbol, board_size=board_size))
    
    return render_template('home.html')

@app.route('/game')
def game():
    # Get data from setup page
    player_symbol = request.args.get('player_symbol')
    board_size = int(request.args.get('board_size'))
    return render_template('game.html', player_symbol=player_symbol, board_size=board_size)

@app.route('/make_move', methods=['POST'])
def make_move():
    # Xử lý yêu cầu từ client, nhận vị trí ô mà người chơi chọn
    data = request.get_json()
    row = data['row']
    col = data['col']
    # Bạn có thể thêm logic xử lý vị trí chơi tại đây
    

    # Phản hồi lại client
    return jsonify({"message": "Move registered", "row": row, "col": col})

