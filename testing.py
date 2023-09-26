import numpy as np
import torch
import pickle

from RL_QLearning.q_learning import QLearning
from models.mrX import MrXModel
from models.detective import DetectiveModel

def test_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device is: {device}")

    interactive = False
    random_start = True
    num_tests = 100
    num_episodes = 20000
    num_detectives = 4
    num_nodes = 21
    max_turns = 15

    mrX_model = None
    mrX_model = MrXModel(num_nodes, num_detectives, device).to(device)
    #mrX_model.restore(episode=num_episodes)
    #mrX_model.eval()
    detectives_model = [DetectiveModel(num_nodes, num_detectives, max_turns, device).to(device) for _ in range(num_detectives)]
    # for i in range(num_detectives):
    #     detectives_model[i].restore(episode=num_episodes+i)
    #     detectives_model[i].eval()

    countD = 0
    countX = 0
    str1 = ""
    print(f"Testing on {num_tests} runs")
    print("Run\tD_wins\tX_wins\n")
    with open('qtables_exponential.pickle', 'rb') as f:
        q_tables = pickle.load(f)
    for i in range(num_tests):
        if interactive:
            print(f"EPISODE {i+1}")
        rl_setting = QLearning(mrX_model, detectives_model, max_turns, explore=0., start=random_start, interact=interactive, q_table = q_tables)
        final_reward, mrX_model, detectives_models, q_table, episode_reward = rl_setting.run_episode(evaluation=True)
        if final_reward < 0:
            countX += 1
        else:
            countD += 1
        str1 = str1 + (str(i+1) + "\t" + str(countD) + "\t" + str(countX))
        str1 = str1 + '\n'
    print(str1)
    print("Detectives =", round(100*countD/num_tests, 2), "%")
    print("Mr.X =", round(100*countX/num_tests, 2), "%")

if __name__ == '__main__':
    test_agent()