# f = open("data/matrix.txt", "r", encoding="utf8")
# content = f.read()
# f.close()
# A = []
# rows = content.split('\n')
# size = len(rows)
# for r in rows:
#   A.append([int(c) for c in r if c != ' '])
size = 21

cart = {(i+1):set() for i in range(size)}
f = open("data/cart.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split('\n'):
  l = s.split(' ')
  cart[int(l[0])] = {int(p) for p in l[1:]}

tram = {(i+1):set() for i in range(size)}
f = open("data/tram.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split('\n'):
  l = s.split(' ')
  tram[int(l[0])] = {int(p) for p in l[1:]}

boat = {(i+1):set() for i in range(size)}
f = open("data/boat.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split('\n'):
  l = s.split(' ')
  boat[int(l[0])] = {int(p) for p in l[1:]}

E = {(i+1):((cart[i+1].union(tram[i+1])).union(boat[i+1])) for i in range(size)}

def moves(place):
  m = ['cart']
  if tram[place]:
    m.append('tram')
  if boat[place]:
    m.append('boat')
  return m

def restrictedMoves(state):
  m = []
  if cart[state[2]].difference(state[1]):
    m.append('cart')
  if tram[state[2]].difference(state[1]):
    m.append('tram')
  if boat[state[2]].difference(state[1]):
    m.append('boat')
  return m

def dest(place, move):
  if move == 'cart':
    return cart[place]
  if move == 'tram':
    return tram[place]
  if move == 'boat':
    return boat[place]

def propagate(source, move):
  target = set()
  for p in source[0]:
    target = target.union(dest(p, move))
  return target.difference(source[1])

def found(state):
  return state[2] in state[1]
