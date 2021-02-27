# %%

import torch
import numpy as np
import agent


class Model(torch.nn.Module):
    def __init__(self, size=3, n_dim=2):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(size ** n_dim, size ** n_dim * 30)
        self.fc2 = torch.nn.Linear(size ** n_dim * 30, size ** n_dim * 30)  # workaround for now
        self.fc3 = torch.nn.Linear(size ** n_dim * 30, size ** n_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        x = self.fc2(x)
        x = torch.nn.ReLU()(x)
        x = self.fc3(x)
        x = torch.nn.Softmax(dim=1)(x)
        return x


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
    prev_state = np.reshape(game_instance.board, (1, -1))
    state = torch.tensor(game_instance.board).view(1, -1).type(torch.float)
    # print(state.shape)
    result = model(state).detach().numpy()[0]
    best_move, best_move_tuple = get_best_possible(game_instance, result)
    new_state, reward, is_done, _ = game_instance.step(best_move_tuple)
    return prev_state, reward, best_move, new_state


def loss(states, actions, rewards, model):
    predicts = torch.log(model(states))
    losses = rewards * predicts[np.arange(len(actions)), actions]
    #print(-losses.mean())
    return -losses.mean()


def train_network(model, game_instance, num_of_iterations):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_values = []
    game_records = []
    wins = 0
    draws = 0
    loses = 0
    for i in range(num_of_iterations):
        game_instance.reset()
        game_record = []
        while not game_instance.is_done():
            result = make_move(game_instance, model)
            game_record.append(result)
            if game_instance.is_done():
                break
            # result = make_move(game, agent2)
            prev_state = np.reshape(game_instance.board, (1, -1))
            move = agent.random_policy(game_instance.board, '')
            new_state, reward, is_done, _ = game_instance.step(move)
            # game_record.append((prev_state, -reward, game_instance.board_position_to_index(move))) # not collecting for one model
        # print(game.current_score)

        wins += game_instance.current_score[0]
        loses += game_instance.current_score[1]
        if game_instance.current_score == (0, 0):
            draws += 1
        scaling_coeff = 0.8
        if game_instance.current_score[0] > 0:
            for j in range(len(game_record) - 2, 0, -1):
                record = list(game_record[j]) #fix workaround
                record[1] += game_record[j + 1][1] * scaling_coeff
                game_record[j] = tuple(record)
        elif game_instance.current_score[0] == game_instance.current_score[1]: #draw case
            last_move = list(game_record[-1])
            last_move[1] = 0.5 #draw reward 0.5 for now
            game_record[-1] = tuple(last_move)
            for j in range(len(game_record) - 2, 0, -1):
                record = list(game_record[j]) #fix workaround
                record[1] += game_record[j + 1][1] * scaling_coeff
                game_record[j] = tuple(record)

        # get a look on draw and lose reward
        for j in game_record:
            game_records.append(j)

        if i % 100 == 0 and i != 0:
            optimizer.zero_grad()
            states = []
            actions = []
            rewards = []
            for j in range(100):
                index = np.random.randint(0, len(game_records))
                # print(j[1][0])
                states.append(game_records[index][0][0])  # to check why
                actions.append(game_records[index][2])
                rewards.append(game_records[index][1])
            states = torch.tensor(states).type(torch.float)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards)
            l = loss(states, actions, rewards, model)
            loss_values.append(l.detach().numpy())

            l.backward()
            optimizer.step()
            # print(actions)
            # print(rewards)
        if i % 1000 == 0:
            print(i)
    return model, loss_values, wins, draws, loses

#
# agent1, loss_values, wins, draws, loses = train_network(agent1, game, 5000)
# print(wins, draws, loses)

