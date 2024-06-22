# Import packages.
import cvxpy as cp
import numpy as np
import gurobipy

Nb = 100 # number of blocks
Ng = 10 # number of GPUs
Nm = 20 # maximum number of blocks per GPU

M = 1e7 # big-ass number

c = np.random.rand(Nb,1) # computational cost per block
tau = np.random.rand(Ng,Ng) # cost of communicating between blocks 
tau = np.zeros((Ng,Ng))
tau[5,6] = 10
# a dictionary containing blocks which communicate
comm = {}
for i in range(Nb):
    comm[i] = []
    for i2 in range(6):
        comm[i].append(int(np.random.rand()*Nb))
#print(comm)

# decision variables
x = cp.Variable(Nb*Ng,boolean=True) # stacked x11 x12 .. xNbxNg
C = cp.Variable() # heaviest GPU
kappa = cp.Variable(Nb) # communication cost per block


# need some matrices to make Python suck less
SumGPU = np.zeros((Nb,Nb*Ng))
SumBlock = np.zeros((Nb,Nb*Ng))
CostPerGPU = np.zeros((Ng,Nb*Ng))
CopyBlock = np.zeros((Nb*Ng,Nb))
CommCost = np.zeros((Nb*Ng,Nb*Ng))

for i in range(Nb):
    for j in range(Ng):
        SumGPU[i,i+j*Nb] = 1.0
        CostPerGPU[j,i+j*Nb] = c[i]
        SumBlock[j,i+j*Nb] = 1.0
        CopyBlock[i+j*Nb,i] = 1.0
        for j2 in range(Ng):
            for i2 in comm[i]:
                CommCost[i+j*Nb,i2+Nb*j2] = tau[j,j2]

                

prob = cp.Problem(cp.Minimize(C+kappa@np.ones((Nb,1))),
                  [SumGPU@x == 1,
                   SumBlock@x <= Nm,
                   CostPerGPU@x <= C, 
                   x + CommCost@x <= CopyBlock@kappa + M*(1-x)])



prob.solve(solver=cp.GUROBI)