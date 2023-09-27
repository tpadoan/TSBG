import pickle
from numpy.random import choice

# size of graph
sizeGraph = 21
# maximum number of turns to catch mr.X
maxTurns = 10
# turns when mr.X location is revealed to the detectives
reveals = [5] # [i+1 for i in range(maxTurns)]
# number of detectives
numDetectives = 3
# flag for fixed initial positions of players, only working if numDetectives < 4
fixed = True
# number of tests
numTests = 100

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

def min_shortest_path(state, node):
  dist = 0
  succ = set(state[0])
  while node not in succ:
    for n in succ:
      succ = succ.union(dest(n))
    dist += 1
  return dist

def mrXmove(state):
  max_dist = 0
  best = 0
  actions = dest(state[1])
  for act in actions:
    dist = min_shortest_path(state, act)
    if max_dist < dist:
      best = act
      max_dist = dist
  return best

def run():
  police = None
  if fixed and numDetectives<4:
    police = [20-7*i for i in range(numDetectives)]
  else:
    police = np.random.choice(np.array(range(1, sizeGraph+1)), size=numDetectives, replace=False)
  mrX = None
  if fixed and numDetectives<4:
    mrX = 5
  else:
    mrX = int((input('Mr.X initial location:\t')).strip())
  state = [police, mrX, {mrX:1.0}]
  turn = 0
  found = False
  while turn < maxTurns and not found:
    turn += 1
    mrX = mrXmove(state)
    move = transportFor(state[1], mrX)
    state[1] = mrX
    if turn in reveals:
      state[2] = {state[1]:1.0}
    else:
      state[2] = propagate_prob(state, move)
    if state[1] in state[0]:
      found = True
      break
    for i in range(numDetectives):
      x = choice(list(state[2].keys()), p=list(state[2].values()))
      state[0][i] = P[turn-1][i+1][tuple((state[0][k-1] if k>0 else x for k in range(numDetectives+1)))]
      diff = state[2].pop(state[0][i], False)
      if diff:
        tot = 1.0 - diff
        for node,prob in state[2].items():
          state[2][node] = prob/tot
      if state[1] in state[0]:
        found = True
        break
  return found


wins = 0
print(f"Testing {numTests} runs")
print("Run\tD_wins\tX_wins")
for count in range(numTests):
  if run():
    wins += 1
  print(f"{1+count}\t{wins}\t{1+count-wins}")
