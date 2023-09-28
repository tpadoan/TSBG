import pickle
from numpy.random import choice
import logic

# size of graph
sizeGraph = 21
# maximum number of turns to catch mr.X
maxTurns = 10
# number of detectives
numDetectives = 3
# flag for fixed initial positions of players, only working if numDetectives < 4
fixed = False
# number of tests
numTests = 1000

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
  transport = None
  if move=='cart':
    transport = cart
  elif move=='tram':
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

def mrXmove1(state):
  best = 0
  maxDist = -1
  actions = dest(state[1])
  for act in actions:
    dist = min_shortest_path(state, act)
    if maxDist < dist:
      best = act
      maxDist = dist
  return best

def mrXmove2(state):
  best = 0
  maxSize = -1
  actions = dest(state[1])
  for act in actions:
    size = len(propagate(state, transportFor(state[1], act)))
    if maxSize < size:
      best = act
      maxSize = size
  return best

def mrXmove(state):
  best = 0
  maxDistSize = -1
  actions = dest(state[1])
  for act in actions:
    dist = min_shortest_path(state, act)
    size = len(propagate(state, transportFor(state[1], act)))
    if maxDistSize < dist*size:
      best = act
      maxSize = dist*size
  return best

def run(G):
  mrX = None
  police = None
  if fixed and numDetectives<4:
    mrX = 5
    police = [20-7*i for i in range(numDetectives)]
  else:
    starts = choice(list(range(1, sizeGraph+1)), size=numDetectives+1, replace=False)
    mrX = starts[0]
    police = starts[1:]
  G.initGame(police, mrX)
  state = [police, mrX, {mrX}]
  turn = 0
  stop = False
  found = False
  while turn < maxTurns and not found and not stop:
    turn += 1
    mrX = mrXmove(state)
    move = transportFor(state[1], mrX)
    state[1] = mrX
    state[2] = propagate(state, move)
    if state[1] in state[0]:
      found = True
      break
    state[0] = G.playTurn(move)
    if not len(state[0]):
      stop = True
    if state[1] in state[0]:
      found = True
  return found


if __name__ == '__main__':
  G = logic.game()
  wins = 0
  print(f"Testing {numTests} runs")
  print("Run\tD_wins\tX_wins")
  for count in range(numTests):
    if run(G):
      wins += 1
    print(f"{1+count}\t{wins}\t{1+count-wins}")