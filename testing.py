import numpy as np
import torch
from RL_QLearning.q_learning import QLearning
from models.mrX import MrXModel
from models.detective import DetectiveModel

def test_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device is: {device}")

    interactive = False
    num_tests = 10
    num_episodes = 30000
    num_detectives = 2
    num_nodes = 21
    max_turns = 20

    mrX_model = None
    #mrX_model = MrXModel(num_nodes, num_detectives, device).to(device)
    #mrX_model.restore(episode=num_episodes)
    #mrX_model.eval()
    detectives_model = [DetectiveModel(num_nodes, num_detectives, max_turns, device).to(device) for _ in range(num_detectives)]
    for i in range(len(detectives_model)):
        detectives_model[i].restore(episode=num_episodes+i)
        detectives_model[i].eval()

    countD = 0
    countX = 0
    str1 = ""
    print(f"Testing {num_tests} runs")
    print("Run\tD_wins\tX_wins\n")
    for i in range(num_tests):
        if interactive:
            print(f"EPISODE {i+1}")
        rl_setting = QLearning(mrX_model, detectives_model, max_turns, explore=0., interact=interactive)
        reward, mrX_model, detectives_model = rl_setting.run_episode()
        if reward < 0:
            countX += 1
        else:
            countD += 1
        str1 = str1 + (str(i+1) + '\t' + str(countD) + '\t' + str(countX))
        str1 = str1 + '\n'
    print(str1)
    print("Detectives =", 100*countD/num_tests, "%")
    print("Mr.X =", 100*countX/num_tests, "%")

if __name__ == '__main__':
    test_agent()