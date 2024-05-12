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
    result = [prob.solve(verbose=False, solver=cp.MOSEK), variable_rho.value]
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


def simulation(target_dm, randomCorrelators, pb, minimizer):
    
    #stato CS
    cs_rho=qp.Qobj(np.array(minF(minimizer, randomCorrelators, target_dm, pb)[1]))
    #save the fidelities
    fid=qp.fidelity(cs_rho,qp.Qobj(target_dm))
    
    return fid, cs_rho.full()
	
def adriMain():


	minimizer = 'cs'
	pb = pauli_basis4q()
	totStates = 20
	noises = [0, 0.2]
	randomCorrs = [1,5,10,20,30,40,50,60,64]

	for n in noises:
		mf = []
		mp = []
		for r in randomCorrs:
			randomCorrelators = rand_corr_choice(r)
			fid = []
			pur = []
			for _ in range(totStates):
				target=qp.rand_ket(16)
				target_dm = target*target.dag()
				target_dm = (1-n)*target_dm.full() + n*np.eye(16)/16
				ff, csState = simulation(target_dm, randomCorrelators, pb,minimizer)
				fid.append(ff)
				pur.append(np.real(np.trace(csState@csState)))
			
			print('FID for '+str(r)+'rands Correlatr is :', np.mean(fid)) 
			print('PUR for '+str(r)+'rands Correlator is :', np.mean(pur)) 
			mf.append(np.mean(fid))
			mp.append(np.mean(pur))
			
		np.savetxt(minimizer+"noise"+str(n)+"-Adrifidelities.txt",mf)
		np.savetxt(minimizer+"noise"+str(n)+"-Adripurity.txt", mp)

if __name__ == "__main__":
	adriMain()