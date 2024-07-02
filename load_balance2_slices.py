# Import packages.
import cvxpy as cp
import numpy as np
import gurobipy


Nm = 60 # maximum number of blocks per GPU

M = 1e7 # big-ass number

ngpu_perslice = np.load('initialdata/nr_gpus_slice.npy')
Ns = len(ngpu_perslice) # number of slices

global_ng = 0 # 
gpu_assign = []
cost_per_gpu = []
block_per_gpu = []

cost_var = 0.1 # % variation in load between GPUs
block_var = 0.5 # % variation in the number of blocks per GPU

for s in range(Ns):
    c = np.load('initialdata/cost_slice'+str(s)+'.npy')
    r = np.load('initialdata/x1_slice'+str(s)+'.npy')
    theta = np.load('initialdata/x2_slice'+str(s)+'.npy')
    
    Nb = len(c) # number of blocks
    Ng = ngpu_perslice[s] # number of GPUs



    # decision variables
    x = cp.Variable(Nb*Ng,boolean=True) # stacked x11 x12 .. xNbxNg
    #C = cp.Variable() # heaviest GPU
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
    
    Bmap = np.zeros((Nb,Nb*Ng))

    for i in range(Nb):
        for j in range(Ng):
            SumGPU[i,i+j*Nb] = 1.0
            CostPerGPU[j,i+j*Nb] = c[i]
            SumBlock[j,i+j*Nb] = 1.0
            CopyBlock[i+j*Nb,i] = 1.0
            R_pos[i+j*Nb,i+j*Nb] = r[i]
            T_pos[i+j*Nb,i+j*Nb] = theta[i]
            Bmap[i,i+j*Nb] = j

    Cav = np.sum(c)/Ng
    Nav = Nb/Ng

    prob = cp.Problem(cp.Minimize(r_p@np.ones((Ng,1))+r_m@np.ones((Ng,1))),
                      [SumGPU@x == 1,
                       SumBlock@x <= Nav*(1+block_var),
                       SumBlock@x >= Nav*(1-block_var),
                       CostPerGPU@x <= (1+cost_var)*Cav,
                       CostPerGPU@x >= (1-cost_var)*Cav,
                       R_pos@x <= SumBlock.T@r_av + SumBlock.T@r_p + M*(1-x),
                       R_pos@x >= SumBlock.T@r_av - SumBlock.T@r_m - M*(1-x),
                       T_pos@x <= SumBlock.T@theta_av + SumBlock.T@t_p + M*(1-x),
                       T_pos@x >= SumBlock.T@theta_av - SumBlock.T@t_m - M*(1-x),
                       r_p >= 0, r_m >= 0, t_p >= 0, t_m >= 0])



    prob.solve(solver=cp.GUROBI)
    
    gpu = Bmap@x.value
    
    for i in range(Nb):
        gpu_assign.append(int(np.round(gpu[i])+global_ng))
        
    global_ng += Ng
    
    cost_per_gpu += list(CostPerGPU@x.value)
    block_per_gpu += list(SumBlock@x.value)

print('COST')
print(max(cost_per_gpu))
print(min(cost_per_gpu))
print(sum(cost_per_gpu)/len(cost_per_gpu))
print(sum(cost_per_gpu))

print('Blocks')
print(max(block_per_gpu))
print(min(block_per_gpu))
print(sum(block_per_gpu)/len(block_per_gpu))
print(sum(block_per_gpu))


np.save('GPUs_constance.npy',np.array(gpu_assign))
    
    
