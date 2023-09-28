import pickle

# size of graph
sizeGraph = 21
# maximum number of turns to catch mr.X
maxTurns = 10
# number of detectives
numDetectives = 3

boat = {i:[] for i in range(1, sizeGraph+1)}
f = open("data/boat.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split('\n'):
  l = s.split(' ')
  boat[int(l[0])] = [int(p) for p in l[1:]]

tram = {i:[] for i in range(1, sizeGraph+1)}
f = open("data/tram.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split('\n'):
  l = s.split(' ')
  tram[int(l[0])] = [int(p) for p in l[1:]]

cart = {i:[] for i in range(1, sizeGraph+1)}
f = open("data/cart.txt", "r", encoding="utf8")
content = f.read()
f.close()
for s in content.split('\n'):
  l = s.split(' ')
  cart[int(l[0])] = [int(p) for p in l[1:]]

dest = {}
for p in range(1, sizeGraph+1):
  dest[p] = cart[p]+tram[p]+boat[p]

def moves(sub_turn, pos):
  if sub_turn>0:
    return [p for p in dest[pos[sub_turn]] if p not in pos[1:]]
  else:
    return dest[pos[sub_turn]]

def succ(turn, sub_turn, pos):
  if pos[0] in pos[1:]:
    return []
  S = []
  if sub_turn == numDetectives:
    for m in moves(sub_turn, pos):
      S.append((turn+1, 0, tuple((m if j==sub_turn else pos[j] for j in range(numDetectives+1)))))
  else:
    for m in moves(sub_turn, pos):
      S.append((turn, sub_turn+1, tuple((m if j==sub_turn else pos[j] for j in range(numDetectives+1)))))
  return S

# init optimal policy
P = [[{(h,i,j,k):None for h in range(1, sizeGraph+1) for i in range(1, sizeGraph+1) for j in range(1, sizeGraph+1) for k in range(1, sizeGraph+1) if i!=j!=k!=i} for s in range(numDetectives+1)] for t in range(maxTurns)]

# init the value function
V = [[{} for s in range(numDetectives+1)] for t in range(maxTurns+1)]
for h in range(1, sizeGraph+1):
  for i in range(1, sizeGraph+1):
    for j in range(1, sizeGraph+1):
      for k in range(1, sizeGraph+1):
        if i!=j!=k!=i:
          if h==i or h==j or h==k:
            V[maxTurns][0][(h,i,j,k)] = (1,maxTurns)
          else:
            V[maxTurns][0][(h,i,j,k)] = (0,maxTurns)
          for t in range(maxTurns):
            for s in range(numDetectives+1):
              if h==i or h==j or h==k:
                V[t][s][(h,i,j,k)] = (1,t)
              else:
                if moves(s, (h,i,j,k)):
                  V[t][s][(h,i,j,k)] = (-1,t) if s>0 else (2,t)
                else:
                  V[t][s][(h,i,j,k)] = (0,t)

for t in range(maxTurns-1, -1, -1):
  for s in range(numDetectives, -1, -1):
    for h in range(1, sizeGraph+1):
      for i in range(1, sizeGraph+1):
        for j in range(1, sizeGraph+1):
          for k in range(1, sizeGraph+1):
            if i!=j!=k!=i:
              for nt,ns,m in succ(t,s,(h,i,j,k)):
                if s>0:
                  if V[nt][ns][m][0]>V[t][s][(h,i,j,k)][0] or (V[nt][ns][m][0]==1 and V[nt][ns][m][1]<V[t][s][(h,i,j,k)][1]):
                    V[t][s][(h,i,j,k)] = V[nt][ns][m]
                    P[t][s][(h,i,j,k)] = m[s]
                else:
                  if V[nt][ns][m][0]<V[t][s][(h,i,j,k)][0] or (V[nt][ns][m][0]==V[t][s][(h,i,j,k)][0] and V[nt][ns][m][1]>V[t][s][(h,i,j,k)][1]):
                    V[t][s][(h,i,j,k)] = V[nt][ns][m]
                    P[t][s][(h,i,j,k)] = m[s]

# stores the optimal policy P on file Pi
pickle.dump(P, open("models/Pi", "wb"))

# to load value function V and optimal policy P from files, do:
# P = pickle.load(open("models/Pi", "rb"))
