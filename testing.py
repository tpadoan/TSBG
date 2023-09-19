import numpy as np
import torch
from RL_QLearning.q_learning import QLearning
from models.mrX import MrXModel
from models.detective import DetectiveModel

def test_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device is: {device}")

    num_episodes = 20000
    interactive = True

    mrX_model = MrXModel(device).to(device)
    mrX_model.restore(episode=num_episodes)
    mrX_model.eval()
    detectives_model = [DetectiveModel(device).to(device) for _ in range(2)]
    for i in range(len(detectives_model)):
        detectives_model[i].restore(episode=num_episodes+i)
        detectives_model[i].eval()
    epsilon = np.linspace(1, 0, num=10)

    countD = 0
    countX = 0
    str1 = ""
    print("Testing")
    print("Run\tD_wins\tX_wins\n")
    for i in range(len(epsilon)):
        if interactive:
            print(f"EPISODE {i+1}")
        rl_setting = QLearning(mrX_model, detectives_model, explore=0., interact=interactive)
        reward, mrX_model, detectives_model = rl_setting.run_episode()
        if reward < 0:
            countX+=1
        else:
            countD+=1
        #if(i%1==0):
            #wins.write(str(i)+"\t"+str(countD)+"\t"+str(countX)+"\n")
        str1 = str1 + (str(round(epsilon[i],2))+'\t'+str(countD)+'\t'+str(countX))
        str1 = str1+'\n'
        #diff[i] = countD-countX
    print(str1)
    print("X=",countX)
    print("Detectives=", countD)

if __name__ == '__main__':
    test_agent()
