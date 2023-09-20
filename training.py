import numpy as np
import torch 

from RL_QLearning.q_learning import QLearning
from models.mrX import MrXModel
from models.detective import DetectiveModel

def train_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device is: {device}")

    random_start = True
    num_episodes = 20000
    num_detectives = 3
    num_nodes = 21
    max_turns = 20

    #mrX_model = MrXModel(num_nodes, num_detectives, device).to(device)
    mrX_model = None
    detectives_models = [DetectiveModel(num_nodes, num_detectives, max_turns, device).to(device) for _ in range(num_detectives)]
    epsilon = np.linspace(1, 0, num=num_episodes)

    countD = 0
    countX = 0
    str1 = ""
    print(f"Training with {num_episodes} runs")
    print("Run\tD_wins\tX_wins\tDiff\n")
    for i in range(num_episodes):
        rl_setting = QLearning(mrX_model, detectives_models, max_turns, explore = epsilon[i], start=random_start)
        reward, mrX_model, detectives_models = rl_setting.run_episode()
        if reward < 0:
            countX += 1
        else:
            countD += 1
        if i>0 and not ((i+1)%(num_episodes/100)):
            print(str(int((i+1)/(num_episodes/100))) + "%\t" + str(countD) + "\t" + str(countX) + "\t" + str(countD-countX))
    print("Detectives =", 100*countD/num_episodes, "%")
    print("Mr.X =", 100*countX/num_episodes, "%")
    #mrX_model.save(episode=num_episodes)
    for i, detective in enumerate(detectives_models):
        detective.save(episode=num_episodes+i)

if __name__ == '__main__':
    train_agent()