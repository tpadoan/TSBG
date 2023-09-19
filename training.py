import numpy as np
import torch 

from RL_QLearning.q_learning import QLearning
from models.mrX import MrXModel
from models.detective import DetectiveModel

def train_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device is: {device}")

    num_episodes = 20000
    num_detectives = 3
    num_nodes = 21
    max_turns = 10

    #mrX_model = MrXModel(num_nodes, num_detectives, device).to(device)
    mrX_model = None
    detectives_models = [DetectiveModel(num_nodes, num_detectives, max_turns, device).to(device) for _ in range(num_detectives)]
    epsilon = np.linspace(1, 0, num=num_episodes)

    start = 0
    countD = 0
    countX = 0
    str1 = ""
    print("Training")
    print("Run\tD_wins\tX_wins\n")
    for i in range(num_episodes):
        rl_setting = QLearning(mrX_model, detectives_models, explore = epsilon[i])
        reward, mrX_model, detectives_models = rl_setting.run_episode()
        if reward < 0:
            countX += 1
        else:
            countD += 1
        #if(i%1==0):
            #wins.write(str(i) + "\t" + str(countD) + "\t" + str(countX) + "\n")
        str1 = str1 + (str(i+1) + '\t' + str(countD) + '\t' + str(countX))
        str1 = str1 + '\n'
        #diff[i] = countD - countX
        if i>0 and not ((i+1)%(num_episodes/100)):
            print(str(int((i+1)/(num_episodes/100)))+"%")
    print(str1)
    print("X=", countX)
    print("Detectives=", countD)
    #mrX_model.save(episode=num_episodes)
    for i, detective in enumerate(detectives_models):
        detective.save(episode=num_episodes+i)

if __name__ == '__main__':
    train_agent()