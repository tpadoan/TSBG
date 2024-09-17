import matplotlib.pyplot as plt
import pickle
from numpy.random import choice

# size of graph
sizeGraph = 21
# maximum number of turns to catch mr.X
maxTurns = 10
# turns when mr.X location is revealed to the detectives
reveals = []  # use [i+1 for i in range(maxTurns)] for fully observable setting
# number of detectives
numDetectives = 3
# flag for fixed initial positions of players, only working if numDetectives < 4
fixed = False
# list of names of moves, same order as in graph utils
movesNames = ["boat", "tram", "cart"]

coords = {}
f = open("data/coords.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split("\n"):
    l = s.split(" ")
    coords[int(l[0])] = (int(l[1]), int(l[2]))

im = plt.imread("data/graph.png")
plt.ion()


def drawMap(state):
    plt.clf()
    plt.imshow(im)
    plt.axis("off")
    X = []
    Y = []
    for p in state[2].keys():
        X.append(coords[p][0])
        Y.append(coords[p][1])
    plt.plot(X, Y, "o", ms=11, color="none", mec="magenta")
    X = []
    Y = []
    for p in state[0]:
        X.append(coords[p][0])
        Y.append(coords[p][1])
    plt.plot(X, Y, "D", ms=9, color="none", mec="cyan")
    if state[1]:
        plt.plot(
            coords[state[1]][0],
            coords[state[1]][1],
            "*",
            ms=10,
            color="none",
            mec="gold",
        )
    plt.show()


boat = {(i + 1): [] for i in range(sizeGraph)}
f = open("data/boat.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split("\n"):
    l = s.split(" ")
    boat[int(l[0])] = [int(p) for p in l[1:]]

tram = {(i + 1): [] for i in range(sizeGraph)}
f = open("data/tram.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split("\n"):
    l = s.split(" ")
    tram[int(l[0])] = [int(p) for p in l[1:]]

cart = {(i + 1): [] for i in range(sizeGraph)}
f = open("data/cart.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split("\n"):
    l = s.split(" ")
    cart[int(l[0])] = [int(p) for p in l[1:]]

P = pickle.load(open("models/Pi", "rb"))
# numEp = 1000000
# Q = pickle.load(open("models/Q"+str(numEp), "rb"))


def bestAct(Q, state):
    moves = tuple(set(dest(state[2][state[1]])).difference(state[2]))
    if not moves:
        return 0
    best = moves[0]
    m = Q[state][best]
    for act in moves[1:]:
        val = Q[state][act]
        if val > m:
            m = val
            best = act
    return best


def dest(source):
    return boat[source] + tram[source] + cart[source]


def transportFor(source, target):
    if target in boat[source]:
        return "boat"
    if target in tram[source]:
        return "tram"
    return "cart"


def propagate_prob(state, move):
    new = {}
    tot = 0
    for node, prob in state[2].items():
        transport = None
        if move == "cart":
            transport = cart
        elif move == "tram":
            transport = tram
        else:
            transport = boat
        succ = [d for d in transport[node] if d not in state[0]]
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
    return new


def distance(x, y):
    dist = 0
    succ = {y}
    while x not in succ:
        for p in succ:
            succ = succ.union(dest(p))
        dist += 1
    return dist


police = None
if fixed and numDetectives < 4:
    police = [5, 6, 20]
else:
    police = choice(list(range(1, sizeGraph + 1)), size=numDetectives, replace=False)
drawMap([police, 0, {}])

mrX = None
if fixed and numDetectives < 4:
    mrX = 13
else:
    mrX = int((input("Mr.X initial location:\t")).strip())
state = [police, mrX, {mrX: 1.0}]
drawMap(state)

turn = 0
found = False
while turn < maxTurns and not found:
    turn += 1
    print("\nTURN " + str(turn))
    # if input('Type anything if found..\t').strip():
    # found = True
    # move = input('Mr.X moves by:\t').strip()
    # while move not in movesNames:
    # move = input('input error, try again:\t').strip()
    mrX = input("mr.X moves to:\t").strip()
    while (not mrX.isdigit()) or (int(mrX) not in dest(state[1])):
        mrX = input("input error, try again:\t").strip()
    mrX = int(mrX)
    move = transportFor(state[1], mrX)
    state[1] = mrX
    if turn in reveals:
        print("Mr.X location has been revealed")
        state[2] = {state[1]: 1.0}
    else:
        state[2] = propagate_prob(state, move)
    drawMap(state)
    if state[1] in state[0]:
        found = True
        break

    for i in range(numDetectives):
        x = choice(list(state[2].keys()), p=list(state[2].values()))
        detMove = P[turn - 1][i + 1][
            tuple((state[0][k - 1] if k > 0 else x for k in range(numDetectives + 1)))
        ]
        print(
            "det "
            + str(i)
            + ": thinks "
            + str(x)
            + " and moves "
            + str(state[0][i])
            + " -> "
            + str(detMove)
        )
        state[0][i] = detMove
        # state[0][i] = bestAct(Q, (turn-1, i, tuple(state[0])))
        diff = state[2].pop(state[0][i], False)
        if diff:
            tot = 1.0 - diff
            for node, prob in state[2].items():
                state[2][node] = prob / tot
        if state[1] in state[0]:
            found = True
            break
    drawMap(state)

if found:
    print("Game ended, the detectives apprehended Mr.X!")
else:
    print("Game ended, Mr.X has escaped!")
