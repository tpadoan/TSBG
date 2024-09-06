import pickle
from numpy.random import choice

class Game:
  def __init__(self):
    sizeGraph = 21
    self.numDetectives = 3
    self.boat = {(i+1):[] for i in range(sizeGraph)}
    f = open("data/boat.txt", "r", encoding="utf8")
    content = f.read()
    f.close()
    for s in content.split('\n'):
      l = s.split(' ')
      self.boat[int(l[0])] = [int(p) for p in l[1:]]
    self.tram = {(i+1):[] for i in range(sizeGraph)}
    f = open("data/tram.txt", "r", encoding="utf8")
    content = f.read()
    f.close()
    for s in content.split('\n'):
      l = s.split(' ')
      self.tram[int(l[0])] = [int(p) for p in l[1:]]
    self.cart = {(i+1):[] for i in range(sizeGraph)}
    f = open("data/cart.txt", "r", encoding="utf8")
    content = f.read()
    f.close()
    for s in content.split('\n'):
      l = s.split(' ')
      self.cart[int(l[0])] = [int(p) for p in l[1:]]
    self.Pi = pickle.load(open("models/Pi", "rb"))

  def initGame(self, detectives: list[int], mrX: int):
    if not mrX:
      self.state = [detectives[:], {i:1.0/21.0 for i in range(1,22)}]
    else:
      self.state = [detectives[:], {mrX:1.0}]
    self.turn = 0

  def getMrXPos(self):
    return self.state[1].keys()

  def dest(self, source: int):
    return self.boat[source] + self.tram[source] + self.cart[source]

  def canMove(self, det: int):
    flag = False
    for m in self.dest(self.state[0][det]):
      if m not in self.state[0]:
        flag = True
    return flag

  def propagateProb(self, move: str):
    transport = None
    if move=='cart':
      transport = self.cart
    elif move=='tram':
      transport = self.tram
    else:
      transport = self.boat
    new = {}
    tot = 0
    for node,prob in self.state[1].items():
      succ = [d for d in transport[node] if d not in self.state[0]]
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
    self.state[1] = new

  def playTurn(self, mrXmove: str):
    self.propagateProb(mrXmove)
    if not len(self.state[1]):
      return (None, True)
    for i in range(self.numDetectives):
      if self.canMove(i):
        x = choice(list(self.state[1].keys()), p=list(self.state[1].values()))
        self.state[0][i] = self.Pi[self.turn][i+1][tuple((self.state[0][k-1] if k>0 else x for k in range(self.numDetectives+1)))]
        diff = self.state[1].pop(self.state[0][i], False)
        if not len(self.state[1]):
          self.turn += 1
          return (self.state[0][:], True)
        if diff:
          tot = 1.0 - diff
          for node,prob in self.state[1].items():
            self.state[1][node] = prob/tot
    self.turn += 1
    return (self.state[0][:], False)
