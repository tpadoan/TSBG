import numpy as np
import torch
import pickle

from RL_QLearning.q_learning import QLearning
from models.mrX import MrXModel
from models.detective import DetectiveModel

def train_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device is: {device}")

    random_start = True
    num_episodes = 50000
    num_detectives = 3
    num_nodes = 21
    max_turns = 15

    #mrX_model = MrXModel(num_nodes, num_detectives, device).to(device)
    mrX_model = None
    detectives_models = [DetectiveModel(num_nodes, num_detectives, max_turns, device).to(device) for _ in range(num_detectives)]
    # epsilon = np.linspace(1, 0.01, num_episodes)
    epsilon = 1
    alpha = 0.1

    countD = 0
    countX = 0
    print(f"Training with {num_episodes} runs")
    print("Run\tD_wins\tX_wins\tDiff\tD_winrate\n")
    rl_setting = QLearning(mrX_model, detectives_models, max_turns, explore = epsilon, start=random_start)
    _, mrX_model, detectives_models, q_table, _ = rl_setting.run_episode()
    rewards = []
    for i in range(num_episodes):
        rl_setting = QLearning(mrX_model, detectives_models, max_turns, explore = epsilon, start=random_start, q_table=q_table, alpha = alpha)
        final_reward, mrX_model, detectives_models, q_table, episode_reward = rl_setting.run_episode()
        if final_reward < 0:
            countX += 1
        else:
            countD += 1
        if epsilon > 0.01:
            epsilon *= 0.999
        rewards.append(episode_reward)
        # alpha *= 10e-4
        if i>0 and not ((i+1)%(num_episodes/100)):
            print(str(int((i+1)/(num_episodes/100))) + "%\t" + str(countD) + "\t" + str(countX) + "\t" + str(countD-countX) + "\t" + str(round(100*countD/(i+1), 2)) + " %")
            print(f"Mean episode reward in last {num_episodes/100} episodes was: {np.mean(rewards)}")
            rewards = []

    print("Detectives =", round(100*countD/num_episodes, 2), "%")
    print("Mr.X =", round(100*countX/num_episodes, 2), "%")

    # Save the final q_tables as pickle file    
    with open("qtables_exponential.pickle", "wb") as f:
        pickle.dump(q_table, f)

    #mrX_model.save(episode=num_episodes)
    # for i, detective in enumerate(detectives_models):
    #     detective.save(episode=num_episodes+i)

if __name__ == '__main__':
    train_agent()