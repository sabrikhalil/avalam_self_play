# -*- coding: utf-8 -*-
"""
Common definitions for the Avalam players.
Copyright (C) 2010 - Vianney le Clément, UCLouvain
Modified by the teaching team of the course INF8215 - 2022, Polytechnique Montréal

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np 

PLAYER1 = 1
PLAYER2 = -1

initial_board = [ [ 0,  0,  1, -1,  0,  0,  0,  0,  0],
                    [ 0,  1, -1,  1, -1,  0,  0,  0,  0],
                    [ 0, -1,  1, -1,  1, -1,  1,  0,  0],
                    [ 0,  1, -1,  1, -1,  1, -1,  1, -1],
                    [ 1, -1,  1, -1,  0, -1,  1, -1,  1],
                    [-1,  1, -1,  1, -1,  1, -1,  1,  0],
                    [ 0,  0,  1, -1,  1, -1,  1, -1,  0],
                    [ 0,  0,  0,  0, -1,  1, -1,  1,  0],
                    [ 0,  0,  0,  0,  0, -1,  1,  0,  0] ]



rows = 9
columns = 9
max_height = 5
action_size = 9*9*8
max_score = 10  

def create_action_dictionary():
    action_dict = {}
    index = 0
    for row in range(rows):
        for col in range(columns):
            for drow in range(-1, 2):
                for dcol in range(-1, 2):
                    if drow == 0 and dcol == 0:
                        continue
                    new_row = row + drow
                    new_col = col + dcol
                    if 0 <= new_row < rows and 0 <= new_col < columns:
                        action_dict[(row, col, new_row, new_col)] = index
                        index += 1
    return action_dict

action_dict = create_action_dictionary()
index_to_action = {index: action for action, index in action_dict.items()}


class InvalidAction(Exception):

    """Raised when an invalid action is played."""

    def __init__(self, action=None):
        self.action = action
        
def get_encoded_state_(state_array):
    board_size = state_array.shape[0]
    encoded_state = np.zeros((2 * max_height + 1, board_size, board_size))

    for i in range(-max_height, max_height + 1):
        mask = (state_array == i).astype(np.int32)
        encoded_state[i + max_height] = mask

    return encoded_state.astype(np.float32)

def get_decoded_state(encoded_state):
    """
    Get the decoded state for the given encoded_state.
    
    Args:
        encoded_state (np.array): A 3D numpy array of shape (2 * max_height + 1, board_size, board_size) representing the encoded game state.
    
    Returns:
        np.array: A 2D numpy array of shape (board_size, board_size) representing the decoded game state.
    """
    board_size = encoded_state.shape[1]
    decoded_state = np.zeros((board_size, board_size), dtype=np.int32)

    for i in range(-max_height, max_height + 1):
        mask = encoded_state[i + max_height].astype(bool)
        decoded_state[mask] = i

    return decoded_state


def is_tower_movable_array(state_array, i, j):
    """Return wether tower (i,j) is movable"""
    for action in get_tower_actions_array(state_array, i, j):
        return True
    return False


def play_action_array(state_array, action):
    """Play an action if it is valid.

    An action is a 4-uple containing the row and column of the tower to
    move and the row and column of the tower to gobble. If the action is
    invalid, raise an InvalidAction exception. Return self.

    """
    if not is_action_valid_array(state_array,action):
        raise InvalidAction(action)
    i1, j1, i2, j2 = action
    h1 = abs(state_array[i1][j1])
    h2 = abs(state_array[i2][j2])
    if state_array[i1][j1] < 0:
        state_array[i2][j2] = -(h1 + h2)
    else:
        state_array[i2][j2] = h1 + h2
    state_array[i1][j1] = 0
    return state_array

def is_finished_array(state_array):
    """Return whether no more moves can be made (i.e., game finished)."""
    for action in get_actions_array(state_array):
        return False
    return True

def get_actions_array(state_array):
    """Yield all valid actions on this board."""
    for i, j, h in get_towers_array(state_array):
        for action in get_tower_actions_array(state_array, i, j):
            yield action
            
def get_tower_actions_array(state_array, i, j):
    """Yield all actions with moving tower (i,j)"""
    h = abs(state_array[i][j])
    if h > 0 and h < max_height:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                action = (i, j, i+di, j+dj)
                if is_action_valid_array(state_array, action):
                    yield action
                    
def get_towers_array(state_array):
    """Yield all towers.

    Yield the towers as triplets (i, j, h):
    i -- row number of the tower
    j -- column number of the tower
    h -- height of the tower (absolute value) and owner (sign)

    """
    for i in range(rows):
        for j in range(columns):
            if state_array[i][j]:
                yield (i, j, state_array[i][j])
                
def is_action_valid_array(state_array, action):
        """Return whether action is a valid action."""
        try:
            i1, j1, i2, j2 = action
            if i1 < 0 or j1 < 0 or i2 < 0 or j2 < 0 or \
               i1 >= rows or j1 >= columns or \
               i2 >= rows or j2 >= columns or \
               (i1 == i2 and j1 == j2) or (abs(i1-i2) > 1) or (abs(j1-j2) > 1):
                return False
            h1 = abs(state_array[i1][j1])
            h2 = abs(state_array[i2][j2])
            if h1 <= 0 or h1 >= max_height or h2 <= 0 or \
                    h2 >= max_height or h1+h2 > max_height:
                return False
            return True
        except (TypeError, ValueError):
            return False
        
def get_actions_indices_array(state_array, action_dict):
    """Yield all valid actions on this board."""
    for i, j, h in get_towers_array(state_array):
        for action in get_tower_actions_array(state_array, i, j):
            yield action_dict[action]
            
def get_score_array(state_array):
    """Return a score for this board.

    The score is the difference between the number of towers of each
    player. In case of ties, it is the difference between the maximal
    height towers of each player. If self.is_finished() returns True,
    this score represents the winner (<0: red, >0: yellow, 0: draw).

    """
    score = 0
    for i in range(rows):
        for j in range(columns):
            if state_array[i][j] < 0:
                score -= 1
            elif state_array[i][j] > 0:
                score += 1
    if score == 0:
        for i in range(rows):
            for j in range(columns):
                if state_array[i][j] == -max_height:
                    score -= 1
                elif state_array[i][j] == max_height:
                    score += 1
    return score / max_score if score != 0 else 0
               

            
def get_encoded_states(states_array):
    """
    Get the encoded states for the given states_array.
    
    Args:
        states_array (np.array): A 3D numpy array of shape (batch_size, board_size, board_size) representing multiple game states.
    
    Returns:
        np.array: A 4D numpy array of shape (batch_size, 2 * max_height + 1, board_size, board_size) representing the encoded game states.
    """
    batch_size, board_size, _ = states_array.shape
    encoded_states = np.zeros((batch_size, 2 * max_height + 1, board_size, board_size))

    for i in range(-max_height, max_height + 1):
        mask = (states_array == i).astype(np.int32)
        encoded_states[:, i + max_height] = mask

    return encoded_states.astype(np.float32)

