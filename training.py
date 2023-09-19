import numpy as np
import torch 

from RL_QLearning.q_learning import QLearning
from models.mrX import MrXModel
from models.detective import DetectiveModel

def train_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device is: {device}")

    mrX_model = MrXModel(device).to(device)
    detectives_models = [DetectiveModel(device).to(device) for _ in range(2)]
    num_episodes = 20000
    epsilon = np.linspace(1, 0, num=num_episodes)

    countD = 0
    countX = 0
    str1 = ""
    print("Training")
    print("Run\tD_wins\tX_wins\n")
    for i in range(len(epsilon)):
        rl_setting = QLearning(mrX_model, detectives_models, explore = epsilon[i])
        reward, mrX_model, detectives_models = rl_setting.run_episode()
        if reward < 0:
            countX+=1
        else:
            countD+=1
        #if(i%1==0):
            #wins.write(str(i)+"\t"+str(countD)+"\t"+str(countX)+"\n")
        str1 = str1 + (str(round(epsilon[i],2))+'\t'+str(countD)+'\t'+str(countX))
        str1 = str1+'\n'
        #diff[i] = countD-countX
        if i>0 and not ((i+1)%(num_episodes/100)):
            print(str(int((i+1)/(num_episodes/100)))+"%")
    print(str1)
    print("X=",countX)
    print("Detectives=", countD)
    mrX_model.save(episode=num_episodes)
    for i, detective in enumerate(detectives_models):
        detective.save(episode=num_episodes+i)

if __name__ == '__main__':
    train_agent()