import numpy as np
import cvxpy as cp
import qutip as qp
from cvxpy.atoms.norm_nuc import normNuc



# these are some functions

def pauli_basis4q():

    pauli_basis=[qp.identity(2),qp.sigmax(), qp.sigmay(), qp.sigmaz()]
    pauli_basis_4q=[]
    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    pauli_basis_4q.append(qp.tensor(pauli_basis[i1], pauli_basis[i2], pauli_basis[i3], pauli_basis[i4]).full())
    return pauli_basis_4q


def count(r):
    vec=[]
    for i in range(len(r)):
        vec.append(float(r[i]))
    print("number of corr used: {}".format(sum(vec)))
    
    

##  MInIMIZATION FUNCTION for THE COMPRESSED SENSING PROBLEM.
## it takes in input a target matrix, a set of correlators generated beforehand, and the pauli correlators

def minF(opt,corrs_counts, targ_mat, Pbasis):    
    #define pauli matrices for two qubits

    pauli_basis_4q = Pbasis
    variable_rho = cp.Variable((16,16), hermitian=True) #two qubit dm
    
    F=0  #se non faccio niente mi ritorna una rappresentazione cs
    
    # Set the remaining constraints
    constraints = [cp.trace(variable_rho) == 1, variable_rho >> 0] 
    
    if opt=="minF":
        F = cp.real(cp.trace(variable_rho@targ_mat))     
    if opt=="cs":
        F= normNuc(variable_rho)
        # Set the remaining constraints
        constraints = [cp.trace(variable_rho) == 1] #in questo caso non uso positivity 

    # Construct the problem.
    objective = cp.Minimize(F)

    
    
    
    ###impose corr constarints
    for i in range(1,256): #not imposing normalization
        if corrs_counts[i-1]=="1":   
            corr_value=cp.real(cp.trace(pauli_basis_4q[i]@targ_mat))
            constraints.append(cp.real(cp.trace(variable_rho@pauli_basis_4q[i])) == corr_value)   

    prob = cp.Problem(objective, constraints)
    result = [prob.solve(verbose=False, solver=cp.SCS), variable_rho.value]
    #result_rescaled = np.real(pow(result, 1/n)) # rescale fidelity with the number of parties

    return result


def rand_corr_choice(x):    
    stringa=np.random.permutation((x)*[1]+(255-x)*[0])
    #print(stringa)
    #print(type(stringa))
    #print(len(stringa))
    stringa1=""
    for i in range(len(stringa)):
        stringa1=stringa1+str(stringa[i])
    return stringa1


def simulation(target_dm, randomCorrelators, pb):
    
    #stato CS
    cs_rho=qp.Qobj(np.array(minF("cs", randomCorrelators, target_dm, pb)[1]))
    #save the fidelities
    fid=qp.fidelity(cs_rho,qp.Qobj(target_dm))
    
    return fid, cs_rho.full()



# these are the things we move
totStates = 2
totCorrs = 2
noises = [0, 0.05, 0.2]
numRandomCorrs = np.concatenate([np.arange(1, 5, 1), np.arange(5, 100, 5)])


fidelities = np.zeros((totCorrs, totStates, numRandomCorrs.shape[0], 3))
purity = np.zeros((totCorrs, totStates, numRandomCorrs.shape[0], 3))
pb = pauli_basis4q()
for k in range(3):
    for j in range(numRandomCorrs.shape[0]):
        for l in range(totCorrs):
            print(k, numRandomCorrs[j], l)
            randomCorrelators = rand_corr_choice(numRandomCorrs[j])
            #calculate the fid for ''totStates'' states
            for i in range(totStates):
                target=qp.rand_ket(16)
                target_dm = target*target.dag()
                target_dm = (1-noises[k])*target_dm.full() + noises[k]*np.eye(16)/16
                fidelities[l, i, j, k], state  = simulation(target_dm, randomCorrelators, pb)
                purity[l, i, j, k] = np.real(np.trace(state@state))

        
np.savetxt("fidelities.txt", fidelities.reshape(-1, 3))
np.savetxt("purity.txt", purity.reshape(-1, 3))