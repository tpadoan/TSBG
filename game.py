import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from sb3_SY import ScotlandYard, mask_fn

from models.detective import DetectiveModel

# size of graph
sizeGraph = 21
# list of names of moves, same order as in graph utils
movesNames = ['boat', 'tram', 'cart']
# maximum number of turns to catch mr.X
maxTurns = 10
# turns when mr.X location is revealed to the detectives
reveals = [1]+[i for i in range(4, maxTurns+1, 3)]
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

def drawMap(state, player_name):
  plt.clf()
  plt.imshow(im)
  plt.axis('off')
  X = []
  Y = []
  for p in state[3]:
    X.append(coords[p][0])
    Y.append(coords[p][1])
  plt.plot(X, Y, 'o', ms=11, color='red', mec='magenta')
  X = []
  Y = []
  for p in state[0]:
    X.append(coords[p][0])
    Y.append(coords[p][1])
  plt.plot(X, Y, 'D', ms=11, color='blue', mec='cyan')
  if state[1]:
    plt.plot(coords[state[1]][0], coords[state[1]][1], '*', ms=10, color='none', mec='gold')
  plt.title(f'{player_name} is currently playing')
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

def getMoves(police, det_id):
  return [(c, 'cart') for c in cart[police[det_id]] if c not in police] + [(t, 'tram') for t in tram[police[det_id]] if t not in police] + [(b, 'boat') for b in boat[police[det_id]] if b not in police]

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


max_turns = 10
detectives_model = MaskablePPO.load(f"models/SB3_detectives/Masked_PPO_SY_POMDP_500k_{max_turns}turns_{numDetectives}detectives_smartMRX_randomStartEachEpisode")
env_SY = ScotlandYard(random_start=True, num_detectives=numDetectives, max_turns=max_turns)
env = ActionMasker(env_SY, mask_fn)
detectives_model.set_env(env)

mrX = env.starting_nodes[0]
tlog = [[0,0,0]]*maxTurns
state = [env.starting_nodes[1:], mrX, tlog, {mrX}]
drawMap(state, "Omo Vespa")

turn = 0
found = False
while turn < maxTurns and not found:
  turn += 1
  print('\nTURN ' + str(turn))
  if turn in reveals:
    print('Mr.X location will be revealed!')
  drawMap(state, "Omo Vespa")
  plt.pause(2)
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
  drawMap(state, "Omo Vespa")
  plt.pause(2)
  if state[1] in state[0]:
    found = True

  env_SY.turn_sub_counter += 1
  for i in range(numDetectives):
    observation = nodes_ohe(state[3]) if turn in reveals else nodes_ohe({-1}) # + [1 if j==i else 0 for j in range(numDetectives)]
    for ohe in [node_ohe(state[0][j]) for j in range(numDetectives)]:
      observation.extend(ohe)
    action_masks = get_action_masks(env)
    action, _ = detectives_model.predict(observation, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
    drawMap(state, f"Detective {i+1}")
    plt.pause(2)
    state[0][i] = action+1
    state[3].discard(state[0][i])
    drawMap(state, f"Detective {i+1}")
    plt.pause(2)
    if state[1] in state[0]:
      found = True
      break

if found:
  print('Game ended, the detectives apprehended Mr.X!')
else:
  print('Game ended, Mr.X has escaped!')