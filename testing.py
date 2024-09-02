import numpy as np
import torch
import pickle

from RL_QLearning.q_learning import QLearning
from models.mrX import MrXModel
from models.detective import DetectiveModel
from sb3_contrib.ppo_mask import MaskablePPO

from sb3_SY import ScotlandYard
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks


def test_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device is: {device}")

    interactive = False
    random_start = True
    num_tests = 100
    num_episodes = 20000
    num_detectives = 3
    num_nodes = 21
    max_turns = 10

    mrX_model = None
    mrX_model = MrXModel(num_nodes, num_detectives, device).to(device)
    #mrX_model.restore(episode=num_episodes)
    #mrX_model.eval()
    detectives_model = [DetectiveModel(num_nodes, num_detectives, max_turns, device).to(device) for _ in range(num_detectives)]
    # for i in range(num_detectives):
    #     detectives_model[i].restore(episode=num_episodes+i)
    #     detectives_model[i].eval()
    detective = MaskablePPO.load(f"/home/anagen/students/nodm/main_spt9u/units/main/anagen/bassoda/TSBG/models/SB3_detectives/Masked_PPO_SY_NO_OBS_500k_10turns_3detectives_smartMRX_randomStartEachEpisode.zip")
    env = ScotlandYard(
        random_start=random_start, num_detectives=num_detectives, max_turns=max_turns, reveal_every=2
    )
    detective.set_env(env)

    countD = 0
    countX = 0
    num_tests = 1000

    str1 = ""
    print(f"Testing on {num_tests} runs")
    print("Run\tD_wins\tX_wins\n")
    for i in range(num_tests):
        done = False
        obs, _ = env.reset()
        # env.render()
        while not done:
            # print(f"before step {obs}")
            action_masks = get_action_masks(env)
            action, _ = detective.predict(obs, action_masks=action_masks)
            obs, reward, done, truncated, info = env.step(action)
            # env.render()
        if reward < 0:
            countX += 1
        elif reward > 0:
            countD += 1
        str1 = str1 + (str(i + 1) + "\t" + str(countD) + "\t" + str(countX))
        str1 = str1 + "\n"
    print(str1)
    print("Detectives =", round(100 * countD / num_tests, 2), "%")
    print("Mr.X =", round(100 * countX / num_tests, 2), "%")

if __name__ == '__main__':
    test_agent()