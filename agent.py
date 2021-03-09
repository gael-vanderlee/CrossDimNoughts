import numpy as np
import pandas as pd
from tqdm import trange


def check_almost_align(board, s):
    size = board.shape[0]
    for irow in range(size):
        if board[irow, :].sum() == s * (size - 1):
            i_empty = np.where(board[irow, :] == 0)[0].item()
            return (irow, i_empty)
    for icol in range(size):
        if board[:, icol].sum() == s * (size - 1):
            i_empty = np.where(board[:, icol] == 0)[0].item()
            return (i_empty, icol)
    if np.diag(board).sum() == s * (size - 1):
        i_empty = np.where(np.diag(board) == 0)[0].item()
        return (i_empty, i_empty)
    if np.diag(np.rot90(board)).sum() == s * (size - 1):
        for i in range(size):
            if board[i, size - 1 - i] == 0:
                return (i, size - 1 - i)
    return None


def random_policy(board, symbol):
    available_pos = np.array(np.where(board == 0)).T
    return tuple(available_pos[np.random.randint(available_pos.shape[0])])


def linear_policy(board, symbol):
    available_pos = np.array(np.where(board == 0)).T
    return tuple(available_pos[0])


def advanced_static_policy(board, symbol):
    coords = check_almost_align(board, symbol)
    if coords is not None:
        return coords
    coords = check_almost_align(board, -symbol)
    if coords is not None:
        return coords
    available_pos = np.array(np.where(board == 0)).T
    return tuple(available_pos[0])


def advanced_random_policy(board, symbol):
    coords = check_almost_align(board, symbol)
    if coords is not None:
        return coords
    coords = check_almost_align(board, -symbol)
    if coords is not None:
        return coords
    available_pos = np.array(np.where(board == 0)).T
    return tuple(available_pos[np.random.randint(available_pos.shape[0])])


def sarsa(game, agent, opponent_policy, alpha, alpha_factor, gamma, epsilon, epsilon_factor, \
          r_win, r_lose, r_even, r_even2, num_episodes):
    for episode_index in trange(num_episodes):
        alpha *= alpha_factor
        epsilon *= epsilon_factor
        state = game.reset()
        if episode_index % 2 == 1:
            action = opponent_policy(state, game.turn)
            state, _, _, _ = game.step(action)
        action = agent.epsilon_greedy_policy(state, epsilon)
        state_history = [state.copy()]
        action_history = [action]
        while True:
            ################### The agent ignores this happens UNLESS it ends the game
            intermediate_state, reward, done, _ = game.step(action)
            if done:
                if reward > 0:
                    final_reward = r_win
                else:
                    final_reward = r_even
                break
            intermediate_action = opponent_policy(intermediate_state, game.turn)
            ################### ------------------------------------------------
            state, reward, done, _ = game.step(intermediate_action)
            if done:
                if reward > 0:
                    reward = r_lose
                else:
                    reward = r_even2
                break
            action = agent.epsilon_greedy_policy(state, epsilon)
            state_history.append(state.copy())
            action_history.append(action)

        agent.update_Qtable(state_history, action_history, reward, alpha, gamma)


class Agent:

    def __init__(self, size=3, policy=None):
        self.q_array = pd.DataFrame(columns=[str(i) + str(j) for i in range(size) for j in range(size)])
        self.policy = policy
        self.identity = {}
        self.convert_rot90_1 = {}
        self.convert_rot90_2 = {}
        self.convert_rot90_3 = {}
        self.convert_horizontal_axis = {}
        self.convert_vertical_axis = {}
        self.convert_diag = {}
        self.convert_antidiag = {}
        empty = np.zeros([size] * 2, dtype=int)

        for i in range(empty.shape[0]):
            for j in range(empty.shape[0]):
                empty *= 0
                empty[i, j] = 1
                tr_empty = empty
                coords_transform = tuple(np.argwhere(tr_empty == 1)[0])
                self.identity[str(i) + str(j)] = coords_transform

        for i in range(empty.shape[0]):
            for j in range(empty.shape[0]):
                empty *= 0
                empty[i, j] = 1
                tr_empty = np.rot90(empty, 1)
                coords_transform = tuple(np.argwhere(tr_empty == 1)[0])
                self.convert_rot90_1[str(i) + str(j)] = coords_transform

        for i in range(empty.shape[0]):
            for j in range(empty.shape[0]):
                empty *= 0
                empty[i, j] = 1
                tr_empty = np.rot90(empty, 2)
                coords_transform = tuple(np.argwhere(tr_empty == 1)[0])
                self.convert_rot90_2[str(i) + str(j)] = coords_transform

        for i in range(empty.shape[0]):
            for j in range(empty.shape[0]):
                empty *= 0
                empty[i, j] = 1
                tr_empty = np.rot90(empty, 3)
                coords_transform = tuple(np.argwhere(tr_empty == 1)[0])
                self.convert_rot90_3[str(i) + str(j)] = coords_transform

        for i in range(empty.shape[0]):
            for j in range(empty.shape[0]):
                empty *= 0
                empty[i, j] = 1
                tr_empty = empty[::-1]
                coords_transform = tuple(np.argwhere(tr_empty == 1)[0])
                self.convert_horizontal_axis[str(i) + str(j)] = coords_transform

        for i in range(empty.shape[0]):
            for j in range(empty.shape[0]):
                empty *= 0
                empty[i, j] = 1
                tr_empty = empty[:, ::-1]
                coords_transform = tuple(np.argwhere(tr_empty == 1)[0])
                self.convert_vertical_axis[str(i) + str(j)] = coords_transform

        for i in range(empty.shape[0]):
            for j in range(empty.shape[0]):
                empty *= 0
                empty[i, j] = 1
                tr_empty = empty.T
                coords_transform = tuple(np.argwhere(tr_empty == 1)[0])
                self.convert_diag[str(i) + str(j)] = coords_transform

        for i in range(empty.shape[0]):
            for j in range(empty.shape[0]):
                empty *= 0
                empty[i, j] = 1
                tr_empty = np.rot90(empty, 2).T
                coords_transform = tuple(np.argwhere(tr_empty == 1)[0])
                self.convert_antidiag[str(i) + str(j)] = coords_transform

    def update_Qtable(self, state_history, action_history, reward, alpha, gamma):
        code_last_state, code_last_action = self.encode_state_and_action(state_history[-1], action_history[-1])
        try:
            self.q_array.loc[code_last_state]
        except:
            self.q_array.loc[code_last_state] = 0.5
        self.q_array.loc[code_last_state, code_last_action] = reward

        for i in range(len(state_history) - 2, -1, -1):
            state = state_history[i]
            new_state = state_history[i + 1]
            action = action_history[i]
            new_action = action_history[i + 1]
            code_state, code_action = self.encode_state_and_action(state, action)
            try:
                self.q_array.loc[code_state]
            except:
                self.q_array.loc[code_state] = 0.5
            code_new_state, code_new_action = self.encode_state_and_action(new_state, new_action)
            self.q_array.loc[code_state, code_action] = (1 - alpha) * self.q_array.loc[code_state, code_action] \
                                                        + alpha * gamma * self.q_array.loc[
                                                            code_new_state, code_new_action]

    def play_vs_opponent(self, game, symbol):
        state = game.board
        if self.policy is not None:
            return self.policy(state, symbol)
        state_code = self.encode_state(state)
        try:
            self.q_array.loc[state_code]
        except:
            legal_moves = np.argwhere(state == 0)
            return tuple(legal_moves[np.random.randint(legal_moves.shape[0])])
        ###### Find best move
        potential_actions = self.q_array.loc[state_code]
        reference_state = self.decode_one_state(state_code)
        free_spots_scores = []

        if np.prod(state == reference_state):
            for a in potential_actions.index:
                tr_coords = self.identity[a]
                if state[tr_coords] == 0:
                    free_spots_scores.append((tr_coords, potential_actions.loc[a]))

        elif np.prod(state == np.rot90(reference_state, 1)):
            for a in potential_actions.index:
                tr_coords = self.convert_rot90_1[a]
                if state[tr_coords] == 0:
                    free_spots_scores.append((tr_coords, potential_actions.loc[a]))

        elif np.prod(state == np.rot90(reference_state, 2)):
            for a in potential_actions.index:
                tr_coords = self.convert_rot90_2[a]
                if state[tr_coords] == 0:
                    free_spots_scores.append((tr_coords, potential_actions.loc[a]))

        elif np.prod(state == np.rot90(reference_state, 3)):
            for a in potential_actions.index:
                tr_coords = self.convert_rot90_3[a]
                if state[tr_coords] == 0:
                    free_spots_scores.append((tr_coords, potential_actions.loc[a]))

        elif np.prod(state == reference_state[::-1]):
            for a in potential_actions.index:
                tr_coords = self.convert_horizontal_axis[a]
                if state[tr_coords] == 0:
                    free_spots_scores.append((tr_coords, potential_actions.loc[a]))

        elif np.prod(state == reference_state[:, ::-1]):
            for a in potential_actions.index:
                tr_coords = self.convert_vertical_axis[a]
                if state[tr_coords] == 0:
                    free_spots_scores.append((tr_coords, potential_actions.loc[a]))

        elif np.prod(state == reference_state.T):
            for a in potential_actions.index:
                tr_coords = self.convert_diag[a]
                if state[tr_coords] == 0:
                    free_spots_scores.append((tr_coords, potential_actions.loc[a]))

        elif np.prod(state == np.rot90(reference_state, 2).T):
            for a in potential_actions.index:
                tr_coords = self.convert_antidiag[a]
                if state[tr_coords] == 0:
                    free_spots_scores.append((tr_coords, potential_actions.loc[a]))
        else:
            print('Error in play_vs_opponent in Agent !!!')
        max_reward = max(free_spots_scores, key=lambda x: x[1])[1]
        best_actions = [act for act, rew in free_spots_scores if rew == max_reward]
        return best_actions[np.random.randint(len(best_actions))]

    def greedy_policy(self, state):
        """
        Return the next action as a tuple
        """
        code_state = self.encode_state(state)
        try:
            self.q_array.loc[code_state]
        except:
            self.q_array.loc[code_state] = 0.5
        reference_state = self.decode_one_state(code_state)
        legal_actions = []
        actions = self.q_array.loc[code_state]

        if np.prod(state == reference_state):
            for act in actions.index:
                coords_transform = self.identity[act]
                if state[coords_transform] == 0:
                    legal_actions.append((coords_transform, actions.loc[act]))
            max_reward = max(legal_actions, key=lambda x: x[1])[1]
            best_actions = [act_rew[0] for act_rew in legal_actions if act_rew[1] == max_reward]
            return best_actions[np.random.randint(len(best_actions))]

        if np.prod(state == np.rot90(reference_state, 1)):
            for act in actions.index:
                coords_transform = self.convert_rot90_1[act]
                if state[coords_transform] == 0:
                    legal_actions.append((coords_transform, actions.loc[act]))
            max_reward = max(legal_actions, key=lambda x: x[1])[1]
            best_actions = [act_rew[0] for act_rew in legal_actions if act_rew[1] == max_reward]
            return best_actions[np.random.randint(len(best_actions))]

        if np.prod(state == np.rot90(reference_state, 2)):
            for act in actions.index:
                coords_transform = self.convert_rot90_2[act]
                if state[coords_transform] == 0:
                    legal_actions.append((coords_transform, actions.loc[act]))
            max_reward = max(legal_actions, key=lambda x: x[1])[1]
            best_actions = [act_rew[0] for act_rew in legal_actions if act_rew[1] == max_reward]
            return best_actions[np.random.randint(len(best_actions))]

        if np.prod(state == np.rot90(reference_state, 3)):
            for act in actions.index:
                coords_transform = self.convert_rot90_3[act]
                if state[coords_transform] == 0:
                    legal_actions.append((coords_transform, actions.loc[act]))
            max_reward = max(legal_actions, key=lambda x: x[1])[1]
            best_actions = [act_rew[0] for act_rew in legal_actions if act_rew[1] == max_reward]
            return best_actions[np.random.randint(len(best_actions))]

        if np.prod(state == reference_state[::-1]):
            for act in actions.index:
                coords_transform = self.convert_horizontal_axis[act]
                if state[coords_transform] == 0:
                    legal_actions.append((coords_transform, actions.loc[act]))
            max_reward = max(legal_actions, key=lambda x: x[1])[1]
            best_actions = [act_rew[0] for act_rew in legal_actions if act_rew[1] == max_reward]
            return best_actions[np.random.randint(len(best_actions))]

        if np.prod(state == reference_state[:, ::-1]):
            for act in actions.index:
                coords_transform = self.convert_vertical_axis[act]
                if state[coords_transform] == 0:
                    legal_actions.append((coords_transform, actions.loc[act]))
            max_reward = max(legal_actions, key=lambda x: x[1])[1]
            best_actions = [act_rew[0] for act_rew in legal_actions if act_rew[1] == max_reward]
            return best_actions[np.random.randint(len(best_actions))]

        if np.prod(state == reference_state.T):
            for act in actions.index:
                coords_transform = self.convert_diag[act]
                if state[coords_transform] == 0:
                    legal_actions.append((coords_transform, actions.loc[act]))
            max_reward = max(legal_actions, key=lambda x: x[1])[1]
            best_actions = [act_rew[0] for act_rew in legal_actions if act_rew[1] == max_reward]
            return best_actions[np.random.randint(len(best_actions))]

        if np.prod(state == np.rot90(reference_state, 2).T):
            for act in actions.index:
                coords_transform = self.convert_antidiag[act]
                if state[coords_transform] == 0:
                    legal_actions.append((coords_transform, actions.loc[act]))
            max_reward = max(legal_actions, key=lambda x: x[1])[1]
            best_actions = [act_rew[0] for act_rew in legal_actions if act_rew[1] == max_reward]
            return best_actions[np.random.randint(len(best_actions))]
        print('\nError in greedy policy !!!\n')
        print(state, '\n')
        print(reference_state, '\n')
        print(reference_state[::-1][::-1].T)

    def epsilon_greedy_policy(self, state, epsilon):
        """
        Return the next action as a tuple
        """
        if self.q_array.empty or np.random.rand() < epsilon:
            legal_moves = np.argwhere(state == 0)
            size = legal_moves.shape[0]
            random_idx = np.random.randint(size)
            action = tuple(legal_moves[random_idx])
        else:
            action = self.greedy_policy(state)
        return action

    def encode_action(self, action):
        if type(action) is str:
            return action
        code_action = ''
        for i in action:
            code_action += str(i)
        return code_action

    def decode_action(self, code_action):
        return tuple([int(i) for i in code_action])

    def encode_one_state(self, state):
        code_state = ''
        for i in state.flatten():
            code_state += str(i) if i != -1 else '2'
        return code_state

    def generate_all_sym_states(self, state):
        sym_states = [self.encode_one_state(state), self.encode_one_state(np.rot90(state, 1)), \
                      self.encode_one_state(np.rot90(state, 2)), self.encode_one_state(np.rot90(state, 3)), \
                      self.encode_one_state(state[::-1]), self.encode_one_state(state[:, ::-1]), \
                      self.encode_one_state(state.T), self.encode_one_state(np.rot90(state, 2).T)]
        sym_states.sort()
        return sym_states

    def encode_state_and_action(self, state, action):
        empty_state = state * 0
        empty_state[action] = 1
        sym = [(self.encode_one_state(state), tuple(np.argwhere(empty_state == 1)[0])), \
               (self.encode_one_state(np.rot90(state, 1)), tuple(np.argwhere(np.rot90(empty_state, 1) == 1)[0])), \
               (self.encode_one_state(np.rot90(state, 2)), tuple(np.argwhere(np.rot90(empty_state, 2) == 1)[0])), \
               (self.encode_one_state(np.rot90(state, 3)), tuple(np.argwhere(np.rot90(empty_state, 3) == 1)[0])), \
               (self.encode_one_state(state[::-1]), tuple(np.argwhere(empty_state[::-1] == 1)[0])), \
               (self.encode_one_state(state[:, ::-1]), tuple(np.argwhere(empty_state[:, ::-1] == 1)[0])), \
               (self.encode_one_state(state.T), tuple(np.argwhere(empty_state.T == 1)[0])), \
               (self.encode_one_state(np.rot90(state, 2).T), tuple(np.argwhere(np.rot90(empty_state, 2).T == 1)[0]))]
        sym.sort(key=lambda x: x[0])
        code_state = sym[0][0]
        tr_action = sym[0][1]
        code_action = self.encode_action(tr_action)
        return code_state, code_action

    def encode_state(self, state):
        sym_states = self.generate_all_sym_states(state)
        return sym_states[0]

    def decode_one_state(self, code_state):
        flat_list = [0 if elem == '0' else 1 if elem == '1' else -1 for elem in list(code_state)]
        size = int(np.sqrt(len(code_state)))
        state = np.reshape(flat_list, (size, size))
        return state

# %%
