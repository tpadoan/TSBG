#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.random import choice

from logic import Game

# size of graph
sizeGraph = 21
# maximum number of turns to catch mr.X
maxTurns = 10
# number of detectives
numDetectives = 3
# flag for fixed initial positions of players, only working if numDetectives < 4
fixed = False

boat = {(i + 1): [] for i in range(sizeGraph)}
f = open("data/boat.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split("\n"):
    l = s.split(" ")
    boat[int(l[0])] = [int(p) for p in l[1:]]

tram = {(i + 1): [] for i in range(sizeGraph)}
f = open("data/bus.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split("\n"):
    l = s.split(" ")
    tram[int(l[0])] = [int(p) for p in l[1:]]

cart = {(i + 1): [] for i in range(sizeGraph)}
f = open("data/bike.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split("\n"):
    l = s.split(" ")
    cart[int(l[0])] = [int(p) for p in l[1:]]


def dest(source):
    return boat[source] + tram[source] + cart[source]


def transportFor(source, target):
    if target in boat[source]:
        return "boat"
    if target in tram[source]:
        return "bus"
    return "bike"


def propagate(state, move):
    new = set()
    transport = None
    if move == "bike":
        transport = cart
    elif move == "bus":
        transport = tram
    else:
        transport = boat
    for node in state[2]:
        new = new.union([d for d in transport[node] if d not in state[0]])
    return new


def min_shortest_path(state, node):
    dist = 0
    succ = set(state[0])
    while node not in succ:
        for n in succ:
            succ = succ.union(dest(n))
        dist += 1
    return dist


def mrXmove(state, strat):
    best = 0
    if strat == 0:
        maxDist = -1
        actions = dest(state[1])
        for act in actions:
            dist = min_shortest_path(state, act)
            if maxDist < dist:
                best = act
                maxDist = dist
    elif strat == 1:
        maxSize = -1
        actions = dest(state[1])
        for act in actions:
            size = len(propagate(state, transportFor(state[1], act)))
            if maxSize < size:
                best = act
                maxSize = size
    elif strat == 2:
        maxDistSize = -1
        actions = dest(state[1])
        for act in actions:
            dist = min_shortest_path(state, act)
            size = len(propagate(state, transportFor(state[1], act)))
            if maxDistSize < dist * size:
                best = act
                maxDistSize = size
    elif strat == 3:
        maxDistSize = -1
        actions = cart[state[1]]
        for act in actions:
            dist = min_shortest_path(state, act)
            size = len(propagate(state, transportFor(state[1], act)))
            if maxDistSize < dist * size:
                best = act
                maxDistSize = dist * size
        if maxDistSize < 2:
            actions = tram[state[1]]
            for act in actions:
                dist = min_shortest_path(state, act)
                size = len(propagate(state, transportFor(state[1], act)))
                if maxDistSize < dist * size:
                    best = act
                    maxDistSize = dist * size
        if maxDistSize < 2:
            actions = boat[state[1]]
            for act in actions:
                dist = min_shortest_path(state, act)
                size = len(propagate(state, transportFor(state[1], act)))
                if maxDistSize < dist * size:
                    best = act
                    maxDistSize = dist * size
    elif strat == 4:
        actionsCart = cart[state[1]]
        actionsTram = tram[state[1]]
        actionsBoat = boat[state[1]]
        moves = actionsCart + actionsTram + actionsBoat
        prob = []
        for act in actionsCart:
            dist = min_shortest_path(state, act)
            prob.append(dist)
        for act in actionsTram:
            dist = min_shortest_path(state, act)
            prob.append(dist / 2.0)
        for act in actionsBoat:
            dist = min_shortest_path(state, act)
            prob.append(dist / 3.0)
        tot = sum(prob)
        if tot > 0:
            for i in range(len(prob)):
                prob[i] = prob[i] / tot
            best = choice(moves, p=prob)
        else:
            best = moves[0]
    return best


def run(G, mrXstrat, useRL):
    mrX = None
    police = None
    if fixed and numDetectives < 4:
        mrX = 13
        police = [5, 6, 20]
    else:
        police = choice(
            list(range(1, sizeGraph + 1)), size=numDetectives, replace=False
        )
        mrX = 0
        maxDist = -1
        starts = [i for i in range(1, 22)]
        for pos in starts:
            dist = min_shortest_path([police], pos)
            if maxDist < dist:
                mrX = pos
                maxDist = dist
    G.initGame(police, mrX, useRL)
    state = [police, mrX, {mrX}]
    turn = 0
    found = False
    while turn < maxTurns and not found:
        if list(state[0]) != list(G.state[0]):
            print("Error: inconsistent detective state")
        if state[2] != set(G.state[1].keys()):
            print("Error: inconsistent mrX inferred locations")
        if state[1] not in state[2]:
            print("Error: inconsistent mrX position w.r.t. inferred ones")
        turn += 1
        mrX = mrXmove(state, mrXstrat)
        move = transportFor(state[1], mrX)
        state[1] = mrX
        state[2] = propagate(state, move)
        if state[1] in state[0]:
            found = True
            break
        state[0], _ = G.playTurn(move)
        state[2] = state[2].difference(state[0])
        if state[1] in state[0]:
            found = True
    return found


if __name__ == "__main__":
    G = Game()
    numTests = 10000
    useRL = True
    for mrXstrat in range(5):
        wins = 0
        print(
            f"\nTesting {numTests} runs with {'RL' if useRL else 'probabilistic'} detectives, mrX strat {mrXstrat}"
        )
        print("Runs\tD_wins\tX_wins\tD_win%")
        for count in range(numTests):
            if run(G, mrXstrat, useRL):
                wins += 1
        print(f"{numTests}\t{wins}\t{numTests-wins}\t{int(100*wins/numTests)}")
