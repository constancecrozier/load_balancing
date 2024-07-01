# Import packages.
import cvxpy as cp
import numpy as np
import gurobipy

Nb = 300 # number of buses
Ng = 10 # number of generators
Nj = 10 # number of loads
Ne = 250 # number of lines

B = np.random.rand(Nb,Nb)
pj = np.random.rand(Ng,1)

pg_max = [1.0]*Ng
pg_min = [0.]*Ng

theta = cp.Variable(Nb)


prob = cp.Problem(cp.Minimize(C+kappa@np.ones((Nb,1))),
                  [SumGPU@x == 1,
                   SumBlock@x <= Nm,
                   CostPerGPU@x <= C, 
                   x + CommCost@x <= CopyBlock@kappa + M*(1-x)])



prob.solve(solver=cp.GUROBI)