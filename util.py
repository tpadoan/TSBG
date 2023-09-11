def edges(M):
  size = len(M)
  E = {i:set() for i in range(size)}
  for v in range(size):
    E[v] = {i for i in range(size) if M[v][i]}
  return E

def maxDistFrom(E, v):
  V = {v for v in range(len(E))}
  d = 0
  R = {v}
  while not R == V:
    d += 1
    T = set()
    for r in R:
      T = T.union(E[r])
    R = R.union(T)
  return d

def diameter(M):
  E = edges(M)
  dia = 0
  for i in range(len(M)):
    d = maxDistFrom(E, i)
    if d > dia:
      dia = d
  return dia

def avgMinDistBetween(E, U, V):
  d = 0.0
  for v in V:
    dv = len(E)
    for u in U:
      dvu = 0
      R = {u}
      while v not in R:
        dvu += 1
        T = set()
        for r in R:
          T = T.union(E[r])
        R = R.union(T)
      if dvu < dv:
        dv = dvu
    d += dv
  return d/len(V)
