# %%

import torch
import numpy as np
from tqdm import tqdm

import agent


class Model(torch.nn.Module):
    def __init__(self, size=3, n_dim=2):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(size ** n_dim, size ** n_dim * 50)
        self.fc2 = torch.nn.Linear(size ** n_dim * 50, size ** n_dim * 50)  # workaround for now
        self.fc3 = torch.nn.Linear(size ** n_dim * 50, size ** n_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.LeakyReLU()(x)
        x = self.fc2(x)
        x = torch.nn.LeakyReLU()(x)
        x = self.fc3(x)
        x = torch.nn.Softmax(dim=1)(x)
        return x

    def save(self, filepath="neuralAgent_weights"):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath="neuralAgent_weights"):
        self.load_state_dict(torch.load(filepath))

# print("transform ", game.board_position_to_index((2, 1, 1)))

def get_best_possible(game_instance, probs):
    probs = [(probs[i], i) for i in range(probs.shape[0])]
    probs = sorted(probs, reverse=True)
    for j in probs:
        position = j[1]
        resulting_position = game_instance.board_position_to_tuple(position)
        if game_instance.board[resulting_position] == 0:
            return position, resulting_position
    return None


def make_move(game_instance, model):
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    prev_state = np.reshape(game_instance.board, (-1))
    state = torch.tensor(game_instance.board).view(1, -1).type(torch.float).to(device)
    result = model(state).cpu().detach().numpy()[0]
    best_move, best_move_tuple = get_best_possible(game_instance, result)
    new_state, reward, is_done, _ = game_instance.step(best_move_tuple)
    return prev_state, reward, best_move


#
def train_network(model, game_instance, num_of_iterations, batch_size, max_records_size=5000, train_for_second=False):
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    criteria = torch.nn.BCELoss()
    loss_values = []
    game_records_x = []
    game_records_y = []
    wins = 0
    draws = 0
    loses = 0
    batch_size = min(num_of_iterations, batch_size)
    for i in tqdm(range(num_of_iterations)):
        game_instance.reset()
        game_record_x = []
        game_record_y = []
        while not game_instance.is_done():
            result = make_move(game_instance, model)
            game_record_x.append(list(result))
            if game_instance.is_done():
                break
            prev_state = np.reshape(game_instance.board, (-1))
            move = agent.random_policy(game_instance.board, '')
            new_state, reward, is_done, _ = game_instance.step(move)
            game_record_y.append(
                [prev_state, reward, game_instance.board_position_to_index(move)])  # not collecting for one model

        if game_instance.current_score[0] > game_instance.current_score[1]:
            wins += 1
            reward = (1, -1)
        elif game_instance.current_score[0] < game_instance.current_score[1]:
            loses += 1
            reward = (-1, 1)
        else:
            draws += 1
            reward = (0.8, 0.8)

        # propagate reward
        scaling_coeff = 0.8
        game_record_x[-1][1] = reward[0]

        def append_game_to_records(game_record, game_records):
            for k in range(len(game_record) - 2, -1, -1):
                game_record[k][1] += game_record[k + 1][1] * scaling_coeff # doing sum, so we can properly play in multidim
            game_records.extend(game_record)
            if len(game_records) > max_records_size:
                game_records = game_records[(len(game_records) - max_records_size):]  # first elements
            return game_records

        game_records_x = append_game_to_records(game_record_x, game_records_x)
        game_records_y = append_game_to_records(game_record_y, game_records_y)

        def train_step(game_records):
            optimizer.zero_grad()
            states = []
            actions = []
            for j in range(batch_size):
                index = np.random.randint(0, len(game_records))
                states.append(game_records[index][0])  # to check why
                actions_list = np.zeros(game_instance.size ** game_instance.n_dim)
                actions_list[game_records[index][2]] = game_records[index][1]
                actions.append(actions_list)
            states = torch.tensor(states, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.float)
            states = states.to(device)
            actions = actions.to(device)
            loss = criteria(model(states), actions)
            loss.backward()
            loss_value = loss.cpu().detach().numpy()
            loss_values.append(loss_value)
            optimizer.step()
            return loss_value

        if i % batch_size == 0 and i != 0:
            if train_for_second:
                loss = train_step(game_records_y)
            else:
                loss = train_step(game_records_x)

            loss_values.append(loss)
    return model, loss_values, wins, draws, loses


def test_against_random(agent1, game, num_of_iterations=1000):
    wins = 0
    draws = 0
    loses = 0
    for i in tqdm(range(num_of_iterations)):
        game.reset()
        while not game.is_done():
            make_move(game, agent1)
            if game.is_done():
                break
            move = agent.random_policy(game.board, '')
            new_state, reward, is_done, _ = game.step(move)

        if game.current_score[0] > game.current_score[1]:
            wins += 1
        elif game.current_score[0] < game.current_score[1]:
            loses += 1
        else:
            draws += 1
    return wins, draws, loses
