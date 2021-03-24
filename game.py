import gym
import numpy as np
from sympy.utilities.iterables import subsets
from sympy.utilities.iterables import multiset_permutations


class Game(gym.Env):
    def __init__(self, p1, p2, size=3, n_dim=2):
        assert (type(n_dim) is int and n_dim >= 2), "wrong n_dim"
        assert (type(size) is int and size >= 2), "wrong size"
        self.n_dim = n_dim
        self.size = size
        self.p1 = p1
        self.p2 = p2
        self.turn = 1
        self.board = np.zeros([size] * n_dim, dtype=int)
        self.current_score = (0, 0)
        self._max_episode_steps = 1000
        super(Game, self).__init__()

    def simulate_games(self, n=100):
        win_p1_a, win_p2_a, tot_even_a = 0, 0, 0
        win_p1_b, win_p2_b, tot_even_b = 0, 0, 0
        for _ in range(n // 2):
            s1, s2 = self.play_a_game(verbose=False)
            if s1 > s2:
                win_p1_a += 1
            elif s2 > s1:
                win_p2_a += 1
            else:
                tot_even_a += 1
        self.p1, self.p2 = self.p2, self.p1
        for _ in range(n // 2):
            s1, s2 = self.play_a_game(verbose=False)
            if s1 > s2:
                win_p2_b += 1
            elif s2 > s1:
                win_p1_b += 1
            else:
                tot_even_b += 1
        self.p1, self.p2 = self.p2, self.p1
        return win_p1_a, win_p2_a, tot_even_a, win_p1_b, win_p2_b, tot_even_b

    def play_a_game(self, verbose=True):
        self.reset()
        if verbose:
            self.render()
        digits = '1234567890'
        while not self.is_done():
            if self.turn == 1:
                player = self.p1
            else:
                player = self.p2
            if type(player) is not str:
                coords = player.play_vs_opponent(self, self.turn)
                if verbose:
                    print('Agent plays :', coords, '\n')
            else:
                coords_str = input('Coordinates of next move : ')
                print()
                coords = []
                for c in coords_str:
                    if c in digits:
                        coords.append(int(c))
                coords = tuple(coords)
                while len(coords) != self.n_dim or max(coords) >= self.size or not self.is_available(coords):
                    coords_str = input('Position not available, please try another one : ')
                    coords = []
                    for c in coords_str:
                        if c in digits:
                            coords.append(int(c))
                    coords = tuple(coords)
            self.step(coords)
            if verbose:
                self.render()
        if verbose:
            print('Game over. Score :', self.current_score)
            if self.current_score[0] > self.current_score[1]:
                print(self.p1, 'wins !')
            elif self.current_score[1] > self.current_score[0]:
                print(self.p2, 'wins !')
            else:
                print('Even score.')
        return self.current_score

    def is_available(self, position):
        return self.board[position] == 0

    def is_done(self):
        if self.n_dim == 2:
            return sum(self.current_score) != 0 or 0 not in self.board
        return 0 not in self.board

    def reset(self):
        self.turn = 1
        self.current_score = (0, 0)
        self.board *= 0
        return self.board.copy()

    def step(self, position):
        self.board[position] = self.turn
        score_p1, score_p2 = self.score()
        score_p1_diff, score_p2_diff = score_p1 - self.current_score[0], score_p2 - self.current_score[1]
        # update only the score of the player that did the latest move
        if self.turn == 1:
            self.current_score = (score_p1, self.current_score[1])
        else:
            self.current_score = (self.current_score[0], score_p2)
        reward = score_p1_diff if self.turn == 1 else score_p2_diff
        self.turn *= -1
        return self.board, reward, self.is_done(), None

    def render(self):
        visual_board = self.board.copy()
        visual_board = np.where(visual_board == -1, 'O', visual_board)
        visual_board = np.where(visual_board == '1', 'X', visual_board)
        visual_board = np.where(visual_board == '0', '.', visual_board)
        for icol in range(self.size):
            for row in visual_board[icol, :]:
                print(row, end=' ')
            print()
        print()

    def score(self):
        score_p1 = 0
        score_p2 = 0

        def slice_to_mask(L):
            """
            Enables to use slicing operator like array[x, y, :, z] with choosing the position
            of the symbol ':' (represented with a -1 instead). For example L can be equal to
            [0, 0, -1, 0] if we want to access self.board[0, 0, :, 0]
            """
            mask = np.zeros([self.size] * self.n_dim, dtype=bool)
            dim = L.index(-1)
            for tile in range(self.size):
                L[dim] = tile
                mask[tuple(L)] = True
            return mask

        # vertical and horizontal axis
        all_axis = []
        for d in range(self.size ** self.n_dim):
            all_axis.append([(d // self.size ** k) % self.size for k in range(self.n_dim)[::-1]])
            # example in 3D case with size 3 :
            # all_axis = [ [i, j, k] for i = 0, 1, 2 for j = 0, 1, 2 for k = 0, 1, 2 ]
        for d in range(self.n_dim):
            d_axis = np.array(all_axis)
            d_axis[:, d] = -1
            d_axis = np.unique(d_axis, axis=0)
            for axis in d_axis:
                space_mask = slice_to_mask(list(axis))
                in_game_axis = self.board[space_mask]
                axis_value = in_game_axis.sum().item()
                if axis_value == self.size:
                    score_p1 += 1
                elif axis_value == -self.size:
                    score_p2 += 1

        # diagonal axis
        diag = np.array([range(self.size)]).T
        antidiag = np.array([range(self.size - 1, -1, -1)]).T
        poss_diag = np.array([diag, antidiag])
        poss_index = list(range(self.size))
        coords_to_check = set()
        for dof in range(self.n_dim - 2, -1, -1):
            dof_fc = self.n_dim - dof
            cpt = 0
            for fc in subsets(poss_diag, dof_fc, repetition=True):
                if cpt == int(dof_fc / 2) + 1:
                    break
                cpt += 1
                frozen_comp = np.array(fc).reshape((dof_fc, self.size)).T
                if dof > 0:
                    for free_comp in subsets(poss_index, dof, repetition=True):
                        free_comp_array = np.repeat(np.array([free_comp]), self.size, axis=0)
                        coords = np.hstack((free_comp_array, frozen_comp))
                        for perm in multiset_permutations(coords.T.tolist()):
                            perm_coords = [list(i) for i in zip(*perm)]
                            perm_coords.sort()
                            coords_to_check.add(tuple(map(tuple, perm_coords)))
                else:
                    coords = frozen_comp
                    for perm in multiset_permutations(coords.T.tolist()):
                        perm_coords = [list(i) for i in zip(*perm)]
                        perm_coords.sort()
                        coords_to_check.add(tuple(map(tuple, perm_coords)))

        for coords in coords_to_check:
            total = 0
            for tile in coords:
                total += self.board[tile]
            if abs(total) == self.size:
                if total > 0:
                    score_p1 += 1
                else:
                    score_p2 += 1

        return score_p1, score_p2

    def almost_align(self):

        def slice_to_mask(L):
            """
            Enables to use slicing operator like array[x, y, :, z] with choosing the position
            of the symbol ':' (represented with a -1 instead). For example L can be equal to
            [0, 0, -1, 0] if we want to access self.board[0, 0, :, 0]
            """
            mask = np.zeros([self.size] * self.n_dim, dtype=bool)
            dim = L.index(-1)
            for tile in range(self.size):
                L[dim] = tile
                mask[tuple(L)] = True
            return mask

        # vertical and horizontal axis
        all_axis = []
        for d in range(self.size ** self.n_dim):
            all_axis.append([(d // self.size ** k) % self.size for k in range(self.n_dim)[::-1]])
            # example in 3D case with size 3 :
            # all_axis = [ [i, j, k] for i = 0, 1, 2 for j = 0, 1, 2 for k = 0, 1, 2 ]
        for d in range(self.n_dim):
            d_axis = np.array(all_axis)
            d_axis[:, d] = -1
            d_axis = np.unique(d_axis, axis=0)
            for axis in d_axis:
                space_mask = slice_to_mask(list(axis))
                in_game_axis = self.board[space_mask]
                axis_value = in_game_axis.sum().item()
                if axis_value == self.size - 1 and -1 not in in_game_axis:
                    for coords in np.argwhere(space_mask == True):
                        if self.board[tuple(coords)] == 0:
                            return tuple(coords)
                elif axis_value == -self.size + 1 and 1 not in in_game_axis:
                    for coords in np.argwhere(space_mask == True):
                        if self.board[tuple(coords)] == 0:
                            return tuple(coords)

        # diagonal axis
        diag = np.array([range(self.size)]).T
        antidiag = np.array([range(self.size - 1, -1, -1)]).T
        poss_diag = np.array([diag, antidiag])
        poss_index = list(range(self.size))
        coords_to_check = set()
        for dof in range(self.n_dim - 2, -1, -1):
            dof_fc = self.n_dim - dof
            cpt = 0
            for fc in subsets(poss_diag, dof_fc, repetition=True):
                if cpt == int(dof_fc / 2) + 1:
                    break
                cpt += 1
                frozen_comp = np.array(fc).reshape((dof_fc, self.size)).T
                if dof > 0:
                    for free_comp in subsets(poss_index, dof, repetition=True):
                        free_comp_array = np.repeat(np.array([free_comp]), self.size, axis=0)
                        coords = np.hstack((free_comp_array, frozen_comp))
                        for perm in multiset_permutations(coords.T.tolist()):
                            perm_coords = [list(i) for i in zip(*perm)]
                            perm_coords.sort()
                            coords_to_check.add(tuple(map(tuple, perm_coords)))
                else:
                    coords = frozen_comp
                    for perm in multiset_permutations(coords.T.tolist()):
                        perm_coords = [list(i) for i in zip(*perm)]
                        perm_coords.sort()
                        coords_to_check.add(tuple(map(tuple, perm_coords)))

        for coords in coords_to_check:
            total_pos, total_neg = 0, 0
            for tile in coords:
                if self.board[tile] == 1:
                    total_pos += 1
                elif self.board[tile] == -1:
                    total_neg += 1
            if total_pos == self.size - 1 and total_neg == 0:
                for tile in coords:
                    if self.board[tile] == 0:
                        return tile
            elif total_neg == self.size - 1 and total_pos == 0:
                for tile in coords:
                    if self.board[tile] == 0:
                        return tile
        return None

    # these methods will stay here as far as they are already in use
    def board_position_to_tuple(self, pos):
        resulting_position = []
        for k in range(self.n_dim):
            resulting_position.insert(0, pos % self.size)
            pos //= self.size
        return tuple(resulting_position)

    def board_position_to_index(self, pos):
        # For dimension >= 3, we enumerate from higher, so this trick will work
        res = 0
        for i in range(len(pos)):
            res += pos[i] * (self.size ** (len(pos) - 1 - i))
        return res

