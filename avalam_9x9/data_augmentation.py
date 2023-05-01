import numpy as np 
from avalam_9x9 import * 

def rotate_board_and_actions(board, actions):
    def rotate_action(action, board_size, angle):
        i, j, k, l = action
        if angle == 270:
            return j, board_size - 1 - i, l, board_size - 1 - k
        elif angle == 180:
            return board_size - 1 - i, board_size - 1 - j, board_size - 1 - k, board_size - 1 - l
        elif angle == 90:
            return board_size - 1 - j, i, board_size - 1 - l, k

    board_size = len(board)
    rotated_boards_and_actions = []

    actions.sort()  # Sort the original actions before rotating

    for angle in [90, 180, 270]:
        rotated_board = np.rot90(board, k=angle // 90)
        rotated_actions = [rotate_action(action, board_size, angle) for action in actions]
        rotated_boards_and_actions.append((rotated_board, rotated_actions))

    return rotated_boards_and_actions




def generate_combinations(state, action_probs, value):
    def rotate_action(action, board_size, angle):
        i, j, k, l = action
        if angle == 0:
            return i, j, k, l
        elif angle == 270:
            return j, board_size - 1 - i, l, board_size - 1 - k
        elif angle == 180:
            return board_size - 1 - i, board_size - 1 - j, board_size - 1 - k, board_size - 1 - l
        elif angle == 90:
            return board_size - 1 - j, i, board_size - 1 - l, k

    board_size = len(state)
    combinations = []

    for angle in [0, 90, 180, 270]:
        for flip_h in [False, True]:
            for flip_v in [False, True]:
                rotated_state = np.rot90(state, k=angle // 90)
                if flip_h:
                    rotated_state = np.fliplr(rotated_state)
                if flip_v:
                    rotated_state = np.flipud(rotated_state)

                rotated_actions_probs = []
                for action_index, prob in enumerate(action_probs):
                    if prob > 0:
                        action = index_to_action[action_index]
                        rotated_action = rotate_action(action, board_size, angle)
                        if flip_v:
                            rotated_action = (board_size - 1 - rotated_action[0], rotated_action[1], board_size - 1 - rotated_action[2], rotated_action[3])
                        if flip_h:
                            rotated_action = (rotated_action[0], board_size - 1 - rotated_action[1], rotated_action[2], board_size - 1 - rotated_action[3])

                        rotated_actions_probs.append((rotated_action, prob))

                rotated_actions_probs.sort(key=lambda x: x[0])
                rotated_action_probs = np.zeros_like(action_probs)
                for rotated_action, prob in rotated_actions_probs:
                    rotated_action_index = action_dict[rotated_action]
                    rotated_action_probs[rotated_action_index] = prob

                encoded_rotated_state = get_encoded_state_(rotated_state)
                combination = (encoded_rotated_state, rotated_action_probs, value)
                combinations.append(combination)

    return combinations
