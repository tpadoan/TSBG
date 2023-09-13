from logic import *
import matplotlib.pyplot as plt

# maximum number of turns to catch mr.X
maxTurns = 10
# list of the turns when mr.X location is revealed to everyone, possibly empty
reveals = []

# this must point to the list (as txt) of coordinates of locations in the new map
coords = {}
f = open("data/coords.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split('\n'):
  l = s.split(' ')
  coords[int(l[0])] = (int(l[1]), int(l[2]))

# this must point to the new map
im = plt.imread('data/graph.png')
plt.ion()

def drawMap(state):
  X = []
  Y = []
  for p in state[0]:
    X.append(coords[p][0])
    Y.append(coords[p][1])
  plt.clf()
  plt.imshow(im)
  plt.axis('off')
  plt.plot(X, Y, 'o', ms=11, color='none', mec='magenta')
  X = []
  Y = []
  for p in state[1]:
    X.append(coords[p][0])
    Y.append(coords[p][1])
  plt.plot(X, Y, 'D', ms=9, color='none', mec='cyan')
  if state[2]:
    plt.plot(coords[state[2]][0], coords[state[2]][1], '*', ms=10, color='none', mec='gold')
  plt.show()

turn = 0
drawMap([[],[],0])
mrX = int((input('Mr.X initial location:\t')).strip())
police = []
n = int(input('Number of detectives:\t').strip())
for i in range(n):
  p = int(input('Detective ' + str(i+1) + ' initial location:\t').strip())
  if 0 < p <= size:
    police.append(p)
  else:
    print('input error')
    quit()

# unknown initial location:
# state = [{i+1 for i in range(size)}.difference(police), police, mrX]
# known initial location:
state = [{mrX}, police, mrX]
drawMap(state)

while turn < maxTurns and not found(state):
  turn += 1
  print('\nTURN ' + str(turn))

  move = input('Mr.X moves by:\t').strip()
  if move not in moves(state[2]):
    print('input error')
    quit()
  mrX = int(input('secretly to location:\t').strip())
  if mrX<1 or mrX>size or mrX not in dest(state[2], move):
    print('input error')
    quit()
  state[2] = mrX
  if turn in reveals:
    print('Mr.X location has been revealed')
    state[0] = {state[2]}
  else:
    state[0] = propagate(state, move)
  drawMap(state)

  if not found(state):
    for i in range(n):
      p = int(input('Detective ' + str(i+1) + ' moves to:\t').strip())
      if p<1 or p>size or p not in E[state[1][i]] or p in state[1][0:i]:
        print('input error')
        quit()
      state[1][i] = p
    state[0] = state[0].difference(state[1])
    drawMap(state)

if found(state):
  print('Game ended, the detectives apprehended Mr.X!')
else:
  print('Game ended, Mr.X has escaped!')
