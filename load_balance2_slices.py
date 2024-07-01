# Import packages.
import cvxpy as cp
import numpy as np
import gurobipy

Ns = 31 # number of slices
Nm = 60 # maximum number of blocks per GPU

M = 1e7 # big-ass number

for s in range(Ns):
    c = np.load('initialdata/cost_slice'+str(s)+'.npy')
    r = np.load('initialdata/x1_slice'+str(s)+'.npy')
    theta = np.load('initialdata/x2_slice'+str(s)+'.npy')
    
    Nb = len(c) # number of blocks
    Ng = 12 # number of GPUs



    # decision variables
    x = cp.Variable(Nb*Ng,boolean=True) # stacked x11 x12 .. xNbxNg
    C = cp.Variable() # heaviest GPU
    r_av = cp.Variable(Ng) # average r position of the blocks on the GPU
    r_p = cp.Variable(Ng) # 
    r_m = cp.Variable(Ng) # 
    theta_av = cp.Variable(Ng) # average theta position of the blocks on the GPU
    t_p = cp.Variable(Ng) # 
    t_m = cp.Variable(Ng) # 


    # need some matrices to make Python suck less
    SumGPU = np.zeros((Nb,Nb*Ng))
    SumBlock = np.zeros((Ng,Nb*Ng))
    CostPerGPU = np.zeros((Ng,Nb*Ng))
    CopyBlock = np.zeros((Nb*Ng,Nb))
    #R_pos = np.zeros((Ng,Nb*Ng))
    R_pos = np.zeros((Ng*Nb,Nb*Ng))
    T_pos = np.zeros((Ng*Nb,Nb*Ng))

    for i in range(Nb):
        for j in range(Ng):
            SumGPU[i,i+j*Nb] = 1.0
            CostPerGPU[j,i+j*Nb] = c[i]
            SumBlock[j,i+j*Nb] = 1.0
            CopyBlock[i+j*Nb,i] = 1.0
            R_pos[i+j*Nb,i+j*Nb] = r[i]
            T_pos[i+j*Nb,i+j*Nb] = theta[i]

    C = 1.5*np.sum(c)/Ng                

    prob = cp.Problem(cp.Minimize(r_p@np.ones((Ng,1))+r_m@np.ones((Ng,1))),
                      [SumGPU@x == 1,
                       SumBlock@x <= Nm,
                       CostPerGPU@x <= C,
                       R_pos@x <= SumBlock.T@r_av + SumBlock.T@r_p + M*(1-x),
                       R_pos@x >= SumBlock.T@r_av - SumBlock.T@r_m - M*(1-x),
                       T_pos@x <= SumBlock.T@theta_av + SumBlock.T@t_p + M*(1-x),
                       T_pos@x >= SumBlock.T@theta_av - SumBlock.T@t_m - M*(1-x),
                       r_p >= 0, r_m >= 0, t_p >= 0, t_m >= 0])



    prob.solve(solver=cp.GUROBI)