import pickle
from numpy.random import choice
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_SY import ScotlandYard

MOVES = ["boat", "tram", "cart"]

class Game:
    def __init__(self):
        self.size_graph = 21
        self.numDetectives = 3
        self.boat = {(i+1):[] for i in range(self.size_graph)}
        f = open("data/boat.txt", "r", encoding="utf8")
        content = f.read()
        f.close()
        for s in content.split("\n"):
            l = s.split(" ")
            self.boat[int(l[0])] = [int(p) for p in l[1:]]
        self.tram = {(i+1):[] for i in range(self.size_graph)}
        f = open("data/tram.txt", "r", encoding="utf8")
        content = f.read()
        f.close()
        for s in content.split("\n"):
            l = s.split(" ")
            self.tram[int(l[0])] = [int(p) for p in l[1:]]
        self.cart = {(i+1):[] for i in range(self.size_graph)}
        f = open("data/cart.txt", "r", encoding="utf8")
        content = f.read()
        f.close()
        for s in content.split("\n"):
            l = s.split(" ")
            self.cart[int(l[0])] = [int(p) for p in l[1:]]
        self.Pi = pickle.load(open("models/Pi", "rb"))
        kwargs = {"random_start": True, "num_detectives": self.numDetectives, "max_turns": 10, "reveal_every": 0}
        vec_env = make_vec_env(ScotlandYard, n_envs=1, env_kwargs=kwargs, vec_env_cls=DummyVecEnv)
        self.PiRL = MaskablePPO.load("models/PPO.zip", env=vec_env)
        self.sy_env = ScotlandYard(**kwargs)

    def initGame(self, detectives: list[int], mrX: int, use_RL:bool):
        self.useRL = use_RL
        self.turn = 0
        self.state = [detectives[:], {mrX: 1.0}]
        if self.useRL:
            self.sy_env.reset()
            self.sy_env.init_from_positions(detectives, mrX)

    def getMrXPos(self):
        return self.state[1].keys()

    def dest(self, source: int):
        return self.boat[source] + self.tram[source] + self.cart[source]

    def transport_ohe(self, move):
        return [1 if move == t else 0 for t in MOVES]

    def node_ohe(self, node):
        return [1 if (i+1) == node else 0 for i in range(self.size_graph)]

    def canMove(self, pos: int):
        flag = False
        for m in self.dest(pos):
            if m not in self.state[0]:
                flag = True
        return flag

    def propagateProb(self, move: str):
        transport = None
        if move == "cart":
            transport = self.cart
        elif move == "tram":
            transport = self.tram
        else:
            transport = self.boat
        new = {}
        tot = 0
        for node, prob in self.state[1].items():
            succ = [d for d in transport[node] if d not in self.state[0]]
            size = len(succ)
            for s in succ:
                p = prob / size
                if s in new:
                    new[s] += p
                else:
                    new[s] = p
                tot += p
        for node, prob in new.items():
            new[node] = prob / tot
        self.state[1] = new

    def playTurn(self, mrXmove: str):
        self.propagateProb(mrXmove)
        if not len(self.state[1]):
            return (None, True)
        self.sy_env.turn_sub_counter += 1
        for i in range(self.numDetectives):
            if self.canMove(self.state[0][i]):
                if self.useRL:
                    mrX_ohe = [0]*self.size_graph
                    obs = mrX_ohe
                    for detective_ohe in [self.node_ohe(self.state[0][j]) for j in range(self.numDetectives)]:
                        obs.extend(detective_ohe)
                    obs.extend(self.transport_ohe(mrXmove))
                    action_masks = get_action_masks(self.sy_env)
                    action, _ = self.PiRL.predict(obs, action_masks=action_masks)
                    _, reward, done, truncated, info = self.sy_env.step(action)
                    self.state[0][i] = action+1
                else:
                    x = choice(list(self.state[1].keys()), p=list(self.state[1].values()))
                    self.state[0][i] = self.Pi[self.turn][i+1][tuple((self.state[0][k - 1] if k > 0 else x for k in range(self.numDetectives+1)))]
                diff = self.state[1].pop(self.state[0][i], False)
                if not len(self.state[1]):
                    self.turn += 1
                    return (self.state[0][:], True)
                if diff:
                    tot = 1.0 - diff
                    for node, prob in self.state[1].items():
                        self.state[1][node] = prob / tot
        self.turn += 1
        return (self.state[0][:], len(self.state[1]) == 1 and not self.canMove(list(self.state[1].keys())[0]))