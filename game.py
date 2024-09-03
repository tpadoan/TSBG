import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from sb3_SY import ScotlandYard, mask_fn

from models.detective import DetectiveModel
import pickle
from numpy.random import choice

# size of graph
sizeGraph = 21
# maximum number of turns to catch mr.X
maxTurns = 10
# turns when mr.X location is revealed to the detectives
reveals =  [] # use [i+1 for i in range(maxTurns)] for fully observable setting
# number of detectives
numDetectives = 3
# flag for fixed initial positions of players, only working if numDetectives < 4
fixed = False
# list of names of moves, same order as in graph utils
movesNames = ['boat', 'tram', 'cart']

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
  for p in state[2].keys():
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

P = pickle.load(open("models/Pi", "rb"))
#numEp = 1000000
#Q = pickle.load(open("models/Q"+str(numEp), "rb"))

def bestAct(Q, state):
  moves = tuple(set(dest(state[2][state[1]])).difference(state[2]))
  if not moves:
    return 0
  best = moves[0]
  m = Q[state][best]
  for act in moves[1:]:
    val = Q[state][act]
    if val>m:
      m = val
      best = act
  return best

def dest(source):
  return boat[source] + tram[source] + cart[source]

def transportFor(source, target):
  if target in boat[source]:
    return 'boat'
  if target in tram[source]:
    return 'tram'
  return 'cart'

def propagate_prob(state, move):
  new = {}
  tot = 0
  for node,prob in state[2].items():
    transport = None
    if move=='cart':
      transport = cart
    elif move=='tram':
      transport = tram
    else:
      transport = boat
    succ = [d for d in transport[node] if d not in state[0]]
    size = len(succ)
    for s in succ:
      p = prob/size
      if s in new:
        new[s] += p
      else:
        new[s] = p
      tot += p
  for node,prob in new.items():
    new[node] = prob/tot
  return new

def distance(x, y):
  dist = 0
  succ = {y}
  while x not in succ:
    for p in succ:
      succ = succ.union(dest(p))
    dist += 1
  return dist

device = "cuda" if torch.cuda.is_available() else "cpu"
# police = None
# if fixed and numDetectives<4:
#   police = [20-7*i for i in range(numDetectives)]
# else:
#   police = np.random.choice(np.array(range(1, sizeGraph+1)), size=numDetectives, replace=False)
# drawMap([police,0,0,[]])
# detectives_model = [DetectiveModel(sizeGraph, numDetectives, maxTurns, device).to(device) for _ in range(numDetectives)]

max_turns = 10
detectives_model = MaskablePPO.load(f"models/SB3_detectives/Masked_PPO_SY_POMDP_500k_{max_turns}turns_{numDetectives}detectives_smartMRX_randomStartEachEpisode")
env_SY = ScotlandYard(random_start=True, num_detectives=numDetectives, max_turns=max_turns)
env = ActionMasker(env_SY, mask_fn)
detectives_model.set_env(env)
# for i in range(numDetectives):
#   detectives_model[i].restore(episode=numEpisodes+i)
#   detectives_model[i].eval()

mrX = env.starting_nodes[0]
# if fixed and numDetectives<4:
#   mrX = 5
# else:
#   mrX = int((input('Mr.X initial location:\t')).strip())
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
  # if input('Type anything if found..\t').strip():
    # found = True
  # move = input('Mr.X moves by:\t').strip()
  # while move not in movesNames:
    # move = input('input error, try again:\t').strip()
  drawMap(state, "Omo Vespa")
  plt.pause(2)
  mrX = input('mr.X moves secretly to:\t').strip()
  while (not mrX.isdigit()) or (int(mrX) not in dest(state[1])):
    mrX = input('input error, try again:\t').strip()
  mrX = int(mrX)
  move = transportFor(state[1], mrX)
  state[1] = mrX
  if turn in reveals:
    print('Mr.X location has been revealed')
    state[2] = {state[1]:1.0}
  else:
    state[3] = propagate(state, move)
  drawMap(state, "Omo Vespa")
  plt.pause(2)
  if state[1] in state[0]:
    found = True
    break

  env_SY.turn_sub_counter += 1
  for i in range(numDetectives):
      # for i in range(numDetectives):
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
