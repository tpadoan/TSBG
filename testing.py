import numpy as np
import torch 

from RL_QLearning.q_learning import QLearning
from models.mrX import MrXModel
from models.detective import DetectiveModel

def test_agent():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device is: {device}")

    num_episodes = 20000

    mrX_model = MrXModel(device).to(device)
    mrX_model.restore(episode=num_episodes)
    mrX_model.eval()
    detectives_model = [DetectiveModel(device).to(device) for _ in range(3)]
    for i in range(len(detectives_model)):
        detectives_model[i].restore(episode=num_episodes+i)
        detectives_model[i].eval()
    epsilon = np.linspace(1, 0, num=10)

    start = 0
    countD = 0
    countX = 0
    str1 = ""
    print("Testing")
    print("Run\tD_wins\tX_wins\n")
    for _ in range(len(epsilon)):
        rl_setting = QLearning(mrX_model, detectives_model, explore = 0.)
        reward, mrX_model, detectives_model = rl_setting.run_episode()
        if reward < 0:
            countX+=1
        else:
            countD+=1
        #if(start%1==0):
            #wins.write(str(start)+"\t"+str(countD)+"\t"+str(countX)+"\n")
        str1 = str1 + (str(round(epsilon[start],2))+'\t'+str(countD)+'\t'+str(countX))
        start = start + 1
        str1 = str1+'\n'
        #diff[i] = countD-countX
    print(str1)
    print("X=",countX)
    print("Detectives=", countD)

if __name__ == '__main__':
    test_agent()