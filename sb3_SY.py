import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
import utils.graph_util, utils.detective_util, utils.mrX_util


class ScotlandYard(gym.Env):
    def __init__(
        self,
        random_start: bool = True,
        num_detectives: int = 3,
        max_turns: int = 10,
        reveal_every: int = 0,
    ):
        self.G = utils.graph_util.generate_graph()
        self.num_detectives = num_detectives
        self.max_turns = max_turns

        ### Initialization
        if random_start or self.num_detectives > 3:
            self.starting_nodes = np.random.choice(
                np.array(range(1, self.G.number_of_nodes() + 1)),
                size=1 + self.num_detectives,
                replace=False,
            )
        else:
            self.starting_nodes = [5] + [20 - 7 * i for i in range(self.num_detectives)]

        # Action space
        self.action_space = Discrete(self.G.number_of_nodes(), start=0)

        # State space description
        # The state will be the one hot encoding of mrX position followed by the i-th detective position and the transport that mrX used
        self.observation_space = MultiDiscrete(
            np.array([2] * self.G.number_of_nodes() * (1 + num_detectives) + [3] * num_detectives)
        )
        initial_state = (
            np.array(
                [
                    utils.graph_util.node_one_hot_encoding(
                        node_id=node, num_nodes=self.G.number_of_nodes()
                    )
                    for node in self.starting_nodes
                ]
            )
            .flatten()
            .tolist()
        )
        initial_state += [0, 0, 0]
        self.state = np.array(initial_state)

        self.turn_number = 0
        self.turn_sub_counter = 0
        if reveal_every:
            self.reveals = [i for i in range(0, max_turns, reveal_every)]
        else:
            self.reveals = [0]

        self.detectives = np.array([[0] for _ in range(num_detectives)])
        self.mrX = np.array([self.starting_nodes[0]])
        for i in range(self.num_detectives):
            self.detectives[i] = self.starting_nodes[i + 1]
        self.mrX_transport = np.array([0, 0, 0])

        self.interactive = False
        self.done = False
        self.reward = 0
        self.img = plt.imread("data/graph.png")

    def step(self, action):
        valid_moves = self.get_valid_moves()
        valid_dst = [move[1] for move in valid_moves]

        # If there are no valid destinations, then mrX cannot move anymore and the game is over
        if self.turn_sub_counter == 0 and not valid_dst:
            self.reward = 1
            self.done = True
            return self.state, self.reward, self.done, False, {}

        # When the action leads to an invalid node, ignore the iteration
        ##### SHOULD NEVER HAPPEN SINCE WE ARE MASKING ACTIONS DIRECTLY #####
        if action + 1 not in valid_dst:
            self.reward = -0.5
            self.done = False
            return self.state, self.reward, self.done, False, {}

        else:
            next_node = action + 1
            # Let the corresponding player play
            if self.turn_sub_counter == 0:
                # Heuristic for mrX:
                # - 80% selects farthest node from the ones occupied by the detectives
                # - 20% chooses one of the available nodes at random
                if random.random() > 0.0:
                    # weights = self.exponential_weighted_distance_mrX(valid_moves)
                    # best_move = random.choices(
                    #     valid_moves, weights=weights, k=1
                    # )[0]
                    # self.mrX[0] = best_move[1]
                    # self.mrX_transport = np.array(best_move[2:])
                    # Maximising min distance from detectives
                    max_dist = 0
                    best_action_list = []
                    for i in range(valid_moves.shape[0]):
                        dist = self.min_shortest_path(valid_moves[i][1])
                        if max_dist < dist:
                            best_action_list = []
                            best_action_list.append(i)
                            max_dist = dist
                        elif max_dist == dist:
                            best_action_list.append(i)
                    if len(best_action_list) > 1:
                        weights = self.weight_transport(
                            [valid_moves[j][2:] for j in best_action_list]
                        )
                    else:
                        weights = [1]
                    best_action_idx = random.choices(
                        best_action_list, weights=weights, k=1
                    )[0]
                    self.mrX[0] = valid_moves[best_action_idx][1]
                    self.mrX_transport = np.array(valid_moves[best_action_idx][2:])
                else:
                    chosen_move = random.choices(valid_moves, k=1)[0]
                    self.mrX[0] = chosen_move[1]
                    self.mrX_transport = np.array(chosen_move[2:])
            else:
                self.detectives[self.turn_sub_counter - 1] = next_node

            self.game_step()

            # Take the observation of the new environment
            self.state, _ = self.observe()

            self.turn_sub_counter += 1

            # if evaluation is True:
            #     env.render()

            # One game turn is over
            if self.turn_sub_counter > self.num_detectives:
                self.turn_sub_counter = 0
                self.turn_number += 1

            # In the case one of the player cannot move anymore, we let the other ones play
            self.skip_turn()

            return self.state, self.reward, self.done, False, {}

    def exponential_weighted_distance_mrX(self, valid_moves):
        weights = np.array([self.min_shortest_path(valid_moves[i][1])**4 for i in range(valid_moves.shape[0])])
        return weights / weights.sum()

    def weight_transport(self, transport_list: np.ndarray[int]) -> np.ndarray[float]:
        """ Properly weight transports. Walking has larger weight than taking the bicycle, than taking the boat.

        Args:
            transport_list: List of the one hot encoding of the transports that mrX can take.

        Returns:
            The weighted transport list.
        """
        weights = np.array([1, 2, 4])
        transports_weights = np.array([np.dot(transport, weights) for transport in transport_list])

        return transports_weights / transports_weights.sum()

    def action_masks(self) -> np.ndarray[int]:
        """Generate the masked array for actions to be used at each sub turn.

        Returns:
            np.ndarray[int]: The masked actions array.
        """
        valid_moves = self.get_valid_moves()
        valid_dst = [move[1] for move in valid_moves]
        action_masks = np.zeros((self.G.number_of_nodes(),))
        for valid_node in valid_dst:
            action_masks[valid_node - 1] = 1

        return action_masks

    def game_step(self):
        """Make a step in the environment and check whether the game is finished or not"""
        if self.interactive:
            self.render()
            if input(f"sub_turn {self.turn_sub_counter} done, continue?\t") == "n":
                quit()

        # Check that mrX can still move
        if utils.mrX_util.valid_moves_list(self.G, self.mrX, self.detectives).size == 0:
            self.mrX_can_move = False
        else:
            self.mrX_can_move = True

        # Check that any detectives can still move
        self.detective_can_move = [
            utils.detective_util.valid_moves_list(self.G, self.detectives, i).size != 0
            for i in range(len(self.detectives))
        ]
        all_detectives_cant_move = not any(self.detective_can_move)

        # If that is not the case, then the game is over
        if all_detectives_cant_move:
            self.done = True
            self.reward = -1
            return

        # Check that mrX is not in the same position as one of the detectives
        for detective in self.detectives:
            if detective[0] == self.mrX[0]:
                self.done = True
                self.reward = 1
                return

        # The last scenario is the one in which we reached the maximum number of turns
        if (
            self.turn_number == self.max_turns - 1
            and self.turn_sub_counter == self.num_detectives
        ):
            self.done = True
            self.reward = -1

    def skip_turn(self):
        """Function to let the other players play in the case one cannot move anymore"""
        if self.turn_sub_counter == 0 and not self.mrX_can_move:
            self.turn_sub_counter += 1
            self.game_step()
            if not self.done:
                self.skip_turn()

        elif (
            self.turn_sub_counter > 0
            and not self.detective_can_move[self.turn_sub_counter - 1]
        ):
            self.turn_sub_counter += 1
            if self.turn_sub_counter > self.num_detectives:
                self.turn_sub_counter = 0
                self.turn_sub_counter += 1
            self.game_step()
            if not self.done:
                self.skip_turn()

    def get_valid_moves(self) -> np.ndarray[int]:
        """Array of valid moves to play.

        Returns:
            np.ndarray[int]: The valid moves array.
        """
        if self.turn_sub_counter != 0:
            return utils.detective_util.valid_moves_list(
                self.G, self.detectives, self.turn_sub_counter - 1
            )
        else:
            return utils.mrX_util.valid_moves_list(self.G, self.mrX, self.detectives)

    def observe(self) -> tuple[np.ndarray[int], int]:
        """Observe the environment based on which player is playing.

        Returns:
            tuple[np.ndarray[int], int]: Current state and current turn sub counter.
        """
        # If it is one of the detectives' turn
        if self.turn_sub_counter != 0:
            return (self.observe_as_detective(), self.turn_sub_counter)
        # Otherwise, it is mrX turn
        else:
            return (self.observe_as_mrX(), self.turn_sub_counter)

    def observe_as_mrX(self) -> np.ndarray[int]:
        """The observation from mrX point of view is made of its position, the detectives ones and the turn number.

        Returns:
            np.ndarray[int]: The current observation of mrX.
        """
        # Add mrX current position
        observation = utils.graph_util.node_one_hot_encoding(
            node_id=self.mrX[0] if self.turn_number in self.reveals else -1,
            num_nodes=self.G.number_of_nodes(),
        )
        # Add each detective current position
        for i in range(len(self.detectives)):
            observation.extend(
                utils.graph_util.node_one_hot_encoding(
                    node_id=self.detectives[i][0], num_nodes=self.G.number_of_nodes()
                )
            )

        observation.extend(self.mrX_transport.tolist())

        return np.array(observation)

    def observe_as_detective(self) -> np.ndarray[int]:
        """The observation from detectives point of view.

        Returns:
            np.ndarray[int]: The current observation of the detectives.
        """
        # Add mrX current position only during revealing turns
        observation = utils.graph_util.node_one_hot_encoding(
            node_id=self.mrX[0] if self.turn_number in self.reveals else -1,
            num_nodes=self.G.number_of_nodes(),
        )
        # Add each detective current position
        for i in range(len(self.detectives)):
            observation.extend(
                utils.graph_util.node_one_hot_encoding(
                    node_id=self.detectives[i][0], num_nodes=self.G.number_of_nodes()
                )
            )

        observation.extend(self.mrX_transport.tolist())

        return np.array(observation)

    def shortest_path(self, det_id: int) -> int:
        """Compute the shortest path length between the det_id detective and mrX.

        Args:
            det_id (int): The detective id.

        Returns:
            int: The resulting number of hops.
        """
        return nx.shortest_path_length(self.G, self.detectives[det_id][0], self.mrX[0])

    def min_shortest_path(self, node):
        md = nx.shortest_path_length(self.G, node, self.detectives[0][0])
        for i in range(1, len(self.detectives)):
            d = nx.shortest_path_length(self.G, node, self.detectives[i][0])
            if d < md:
                md = d
        return md

    def init_from_positions(self, detectives: list[int], mrX: int):
        new_start_nodes = [mrX]
        new_start_nodes.extend(detectives)
        self.starting_nodes = np.array(new_start_nodes)

        initial_state = (
            np.array(
                [
                    utils.graph_util.node_one_hot_encoding(
                        node_id=node, num_nodes=self.G.number_of_nodes()
                    )
                    for node in self.starting_nodes
                ]
            )
            .flatten()
            .tolist()
        )
        initial_state += [0, 0, 0]
        self.state = np.array(initial_state)
        self.mrX = np.array([self.starting_nodes[0]])
        for i in range(self.num_detectives):
            self.detectives[i] = self.starting_nodes[i + 1]

    def render(self):
        plt.clf()
        plt.imshow(self.img)
        plt.axis("off")
        X = []
        Y = []
        for v in self.mrX:
            x, y = utils.graph_util.get_coords(self.G, v)
            X.append(x)
            Y.append(y)
        plt.plot(X, Y, "o", ms=11, color="red", mec="magenta")
        X = []
        Y = []
        for v in self.detectives:
            x, y = utils.graph_util.get_coords(self.G, v[0])
            X.append(x)
            Y.append(y)
        plt.plot(X, Y, "D", ms=11, color="blue", mec="cyan")
        plt.title(f"Player {self.turn_sub_counter} turn")
        plt.show()

    def reset(self, seed=None, options=None):
        # Generate a new set of starting nodes
        self.starting_nodes = np.random.choice(
            np.array(range(1, self.G.number_of_nodes() + 1)),
            size=1 + self.num_detectives,
            replace=False,
        )
        self.mrX = np.array([self.starting_nodes[0]])
        self.mrX_transport = np.array([0, 0, 0])
        for i in range(self.num_detectives):
            self.detectives[i] = self.starting_nodes[i + 1]
        self.done = False
        self.reward = 0
        initial_state = (
            np.array(
                [
                    utils.graph_util.node_one_hot_encoding(
                        node_id=node, num_nodes=self.G.number_of_nodes()
                    )
                    for node in self.starting_nodes
                ]
            )
            .flatten()
            .tolist()
        )
        initial_state += [0, 0, 0]
        self.state = np.array(initial_state)
        self.turn_number = 0
        self.turn_sub_counter = 0

        return self.state, {}


def mask_fn(env: ScotlandYard) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_masks()


if __name__ == "__main__":
    train = False
    if train:
        random_start = True
        num_detectives = 3
        num_nodes = 21
        max_turns = 10
        reveal_every = 0
        n_envs = 16
        timesteps = 2**12
        total_steps = 30e6

        obs = "NO_OBS" if reveal_every == 0 else f"OBS_EVERY_{reveal_every}"
        model_name = f"Masked_PPO_SY_{obs}_{(total_steps/1e6):.3f}M_{max_turns}turns_{num_detectives}detectives_smarterMRX_finetuned"
        models_dir = f"/home/anagen/students/nodm/main_spt9u/units/main/anagen/bassoda/TSBG/models/SB3_detectives/{model_name}"

        kwargs = {
            "random_start": random_start,
            "num_detectives": num_detectives,
            "max_turns": max_turns,
            "reveal_every": reveal_every,
        }

        vec_env = make_vec_env(
            ScotlandYard,
            n_envs=n_envs,
            env_kwargs=kwargs,
            vec_env_cls=DummyVecEnv,
        )

        eval_callback = MaskableEvalCallback(
            vec_env,
            best_model_save_path=models_dir,
            eval_freq=timesteps * n_envs,
            deterministic=False,
            render=False,
        )

        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            gamma=0.9995,
            learning_rate=3.7e-4,
            ent_coef=6.453e-3,
            n_steps=timesteps,
            n_epochs=4,
            batch_size=64,
        )

        # model = MaskablePPO.load(f"models/SB3_detectives/Masked_PPO_SY_NO_OBS_500k_{max_turns}turns_{num_detectives}detectives_smartMRX_randomStartEachEpisode")
        # model.set_env(env)
        # # # model = PPO("MlpPolicy", env, verbose=1, n_steps=5000, n_epochs=1)
        # # # model = DQN("MlpPolicy", env, verbose=1)
        # # # # print("Policy results before training")
        # # # # evaluate_policy(model, env, n_eval_episodes=1, render=False)

        model.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=False,
            callback=eval_callback,
        )
        model.save(models_dir + "/final_model.zip")

        # model = MaskablePPO.load(
        #     f"models/SB3_detectives/Masked_PPO_SY_POMDP_500k_{max_turns}turns_{num_detectives}detectives_smartMRX_randomStartEachEpisode"
        # )
        # model = PPO.load(f"PPO_SY_50k_{max_turns}turns_smartMRX_randomStartEachEpisode")
        # model = DQN.load(f"DQN_SY_100k_{max_turns}turns_smartMRX_randomStartEachEpisode")
        # model.set_env(env)

    else:
        random_start = True
        num_detectives = 3
        num_nodes = 21
        max_turns = 10
        reveal_every = 0

        kwargs = {
            "random_start": random_start,
            "num_detectives": num_detectives,
            "max_turns": max_turns,
            "reveal_every": reveal_every,
        }

        vec_env = make_vec_env(
            ScotlandYard,
            n_envs=1,
            env_kwargs=kwargs,
            vec_env_cls=DummyVecEnv,
        )

        test_env = ScotlandYard(
            random_start,
            num_detectives,
            max_turns,
            reveal_every,
        )

        model = MaskablePPO.load(
            "/home/anagen/students/nodm/main_spt9u/units/main/anagen/bassoda/TSBG/models/SB3_detectives/Masked_PPO_SY_OBS_EVERY_3_30.000M_10turns_3detectives_smarterMRX_finetuned/final_model.zip",
            env=vec_env,
        )

        countD = 0
        countX = 0
        num_tests = int(1e5)

        str1 = ""
        print(f"Testing on {num_tests} runs")
        print("Run\tD_wins\tX_wins\n")
        for i in range(num_tests):
            done = False
            obs, _ = test_env.reset()
            # test_env.render()
            while not done:
                # print(f"before step {obs}")
                action_masks = get_action_masks(test_env)
                action, _ = model.predict(obs, action_masks=action_masks)
                obs, reward, done, truncated, info = test_env.step(action)
                # test_env.render()
            if reward < 0:
                countX += 1
            elif reward > 0:
                countD += 1
            str1 = str1 + (str(i + 1) + "\t" + str(countD) + "\t" + str(countX))
            str1 = str1 + "\n"
        print(str1)
        print("Detectives =", round(100 * countD / num_tests, 2), "%")
        print("Mr.X =", round(100 * countX / num_tests, 2), "%")
