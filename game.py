import matplotlib.pyplot as plt
import numpy as np
import torch
from models.detective import DetectiveModel

# size of graph
sizeGraph = 21
# list of names of moves, same order as in graph utils
movesNames = ['boat', 'tram', 'cart']
# turns when mr.X location is revealed to the detectives
reveals = []
# maximum number of turns to catch mr.X
maxTurns = 20
# number of detectives
numDetectives = 3
# flag for fixed initial positions of players, only working if numDetectives < 4
fixed = False
# number of epispdes used for learning
numEpisodes = 20000

coords = {}
f = open("data/coords.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split('\n'):
  l = s.split(' ')
  coords[int(l[0])] = (int(l[1]), int(l[2]))

im = plt.imread('data/graph.png')
plt.ion()

def drawMap(state):
  plt.clf()
  plt.imshow(im)
  plt.axis('off')
  X = []
  Y = []
  for p in state[3]:
    X.append(coords[p][0])
    Y.append(coords[p][1])
  plt.plot(X, Y, 'o', ms=11, color='none', mec='magenta')
  X = []
  Y = []
  for p in state[0]:
    X.append(coords[p][0])
    Y.append(coords[p][1])
  plt.plot(X, Y, 'D', ms=9, color='none', mec='cyan')
  if state[1]:
    plt.plot(coords[state[1]][0], coords[state[1]][1], '*', ms=10, color='none', mec='gold')
  plt.show()

boat = {(i+1):[] for i in range(sizeGraph)}
f = open("data/boat.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split('\n'):
  l = s.split(' ')
  boat[int(l[0])] = [int(p) for p in l[1:]]

tram = {(i+1):[] for i in range(sizeGraph)}
f = open("data/tram.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split('\n'):
  l = s.split(' ')
  tram[int(l[0])] = [int(p) for p in l[1:]]

cart = {(i+1):[] for i in range(sizeGraph)}
f = open("data/cart.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split('\n'):
  l = s.split(' ')
  cart[int(l[0])] = [int(p) for p in l[1:]]

def node_ohe(node):
  return [1 if (i+1) == node else 0 for i in range(sizeGraph)]

def nodes_ohe(nodes):
  return [1 if (i+1) in nodes else 0 for i in range(sizeGraph)]

def transport_ohe(move):
  return [1 if move == t else 0 for t in movesNames]

def getMoves(source):
  return [(c, 'cart') for c in cart[source]] + [(t, 'tram') for t in tram[source]] + [(b, 'boat') for b in boat[source]]

def dest(source):
  return boat[source] + tram[source] + cart[source]

def transportFor(source, target):
  if target in boat[source]:
    return 'boat'
  if target in tram[source]:
    return 'tram'
  return 'cart'

def propagate(state, move):
  new = set()
  for s in state[3]:
    if move == 'boat':
      new = new.union(boat[s])
    elif move == 'tram':
      new = new.union(tram[s])
    else:
      new = new.union(cart[s])
  return new.difference(state[0])

device = "cuda" if torch.cuda.is_available() else "cpu"
police = None
if fixed and numDetectives<4:
  police = [20-7*i for i in range(self.num_detectives)]
else
  police = np.random.choice(np.array(range(1, sizeGraph+1)), size=numDetectives, replace=False)
drawMap([police,0,0,[]])
detectives_model = [DetectiveModel(sizeGraph, numDetectives, maxTurns, device).to(device) for _ in range(numDetectives)]
for i in range(numDetectives):
  detectives_model[i].restore(episode=numEpisodes+i)
  detectives_model[i].eval()

mrX = None
if fixed and numDetectives<4:
  mrX = 5
else:
  mrX = int((input('Mr.X initial location:\t')).strip())
tlog = [[0,0,0]]*maxTurns
state = [police, mrX, tlog, {mrX}]
drawMap(state)

turn = 0
found = False
while turn < maxTurns and not found:
  turn += 1
  print('\nTURN ' + str(turn))
  # if input('Type anything if found..\t').strip():
    # found = True
  # move = input('Mr.X moves by:\t').strip()
  # while move not in movesNames:
    # move = input('input error, try again:\t').strip()
  mrX = input('mr.X moves secretly to:\t').strip()
  while (not mrX.isdigit()) or (int(mrX) not in dest(state[1])):
    mrX = input('input error, try again:\t').strip()
  mrX = int(mrX)
  move = transportFor(state[1], mrX)
  state[1] = mrX
  state[2][turn-1] = transport_ohe(move)
  if turn in reveals:
    print('Mr.X location has been revealed')
    state[3] = {state[1]}
  else:
    state[3] = propagate(state, move)
  drawMap(state)
  if state[1] in state[0]:
    found = True

  for i in range(numDetectives):
    observation = nodes_ohe(state[3]) + [1 if j==i else 0 for j in range(numDetectives)]
    for j in range(numDetectives):
      observation.extend(node_ohe(state[0][j]))
    for t in state[2]:
      observation.extend(t)
    actions = getMoves(state[0][i])
    obs = [[] for _ in range(len(actions))]
    for j in range(len(actions)):
      obs[j] = observation + node_ohe(actions[j][0]) # + transport_ohe(actions[j][1])
    state[0][i] = actions[np.argmax(detectives_model[i].predict(obs))][0]
    state[3].discard(state[0][i])
  drawMap(state)
  if state[1] in state[0]:
    found = True

if found:
  print('Game ended, the detectives apprehended Mr.X!')
else:
  print('Game ended, Mr.X has escaped!')