#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle

from numpy import zeros
from numpy.random import choice
from sb3_contrib.ppo_mask import MaskablePPO


class Game:
    def __init__(self):
        self.size_graph = 21
        self.numDetectives = 3
        self.boat = {(i + 1): [] for i in range(self.size_graph)}
        f = open("data/boat.txt", "r", encoding="utf8")
        content = f.read()
        f.close()
        for s in content.split("\n"):
            l = s.split(" ")
            self.boat[int(l[0])] = [int(p) for p in l[1:]]
        self.tram = {(i + 1): [] for i in range(self.size_graph)}
        f = open("data/bus.txt", "r", encoding="utf8")
        content = f.read()
        f.close()
        for s in content.split("\n"):
            l = s.split(" ")
            self.tram[int(l[0])] = [int(p) for p in l[1:]]
        self.cart = {(i + 1): [] for i in range(self.size_graph)}
        f = open("data/bike.txt", "r", encoding="utf8")
        content = f.read()
        f.close()
        for s in content.split("\n"):
            l = s.split(" ")
            self.cart[int(l[0])] = [int(p) for p in l[1:]]
        self.Pi = pickle.load(open("models/presolved.pickle", "rb"))
        self.PiRL = MaskablePPO.load("models/RLPPO.zip")

    def initGame(self, detectives: list[int], mrX: int, use_RL: bool):
        self.useRL = use_RL
        self.turn = 0
        self.state = [detectives[:], {mrX: 1.0}]
        self.mrX_ohe = self.node_ohe(mrX)

    def getMrXPos(self):
        return self.state[1].keys()

    def dest(self, source: int):
        return self.boat[source] + self.tram[source] + self.cart[source]

    def transport_ohe(self, move):
        return [1 if move == t else 0 for t in ["boat", "bus", "bike"]]

    def node_ohe(self, node):
        return [1 if (i + 1) == node else 0 for i in range(self.size_graph)]

    def canMove(self, pos: int):
        flag = False
        for m in self.dest(pos):
            if m not in self.state[0]:
                flag = True
        return flag

    def propagateProb(self, move: str):
        transport = None
        if move == "bike":
            transport = self.cart
        elif move == "bus":
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
            return None, True
        for i in range(self.numDetectives):
            if self.canMove(self.state[0][i]):
                if self.useRL:
                    obs = [0] * self.size_graph  # no info about mrX location

                    for detective_ohe in [
                        self.node_ohe(self.state[0][j])
                        for j in range(self.numDetectives)
                    ]:
                        obs.extend(detective_ohe)
                    obs.extend(self.transport_ohe(mrXmove))
                    masks = zeros((self.size_graph,))
                    for detMove in self.dest(self.state[0][i]):
                        if detMove not in self.state[0]:
                            masks[detMove - 1] = 1
                    action, _ = self.PiRL.predict(obs, action_masks=masks)  # type: ignore
                    self.state[0][i] = action + 1  # type: ignore
                else:
                    x = choice(
                        list(self.state[1].keys()), p=list(self.state[1].values())
                    )
                    self.state[0][i] = self.Pi[self.turn][i + 1][
                        tuple(
                            (
                                self.state[0][k - 1] if k > 0 else x
                                for k in range(self.numDetectives + 1)
                            )
                        )
                    ]
                diff = self.state[1].pop(self.state[0][i], False)
                if not len(self.state[1]):
                    self.turn += 1
                    return self.state[0][:], True
                if diff:
                    tot = 1.0 - diff
                    for node, prob in self.state[1].items():
                        self.state[1][node] = prob / tot
        self.turn += 1
        return self.state[0][:], len(self.state[1]) == 1 and not self.canMove(
            list(self.state[1].keys())[0]
        )
