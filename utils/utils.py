#projection on the feasible set
import torch
import torch.nn as nn
from numpy import trace
import cvxpy as cp
#import qutip as qp
from scipy.linalg import sqrtm
import qutip as qp
from numpy import delete

#voglio fare che corrs_counts sia una stringa di 0 e 1, che mi dice solo se considerare o meno il correlatore.
def generatePaulimatrices():
    
    #define pauli matrices for two qubits
    pauli_basis=[qp.identity(2),qp.sigmax(), qp.sigmay(), qp.sigmaz()]
    pauli_basis_4q=[]
    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    pauli_basis_4q.append(qp.tensor(pauli_basis[i1], pauli_basis[i2], pauli_basis[i3], pauli_basis[i4]).full())

    return pauli_basis_4q

def proj(corrs_counts, corr_vec, corr_org, tiponorma,pauli_basis_4q):    
 
    
    variable_rho = cp.Variable((16,16), hermitian=True) #four qubit dm
    
    targ_mat=stato_from_corr(corr_vec,pauli_basis_4q) #for F
    #print(targ_mat)
    org_mat=stato_from_corr(corr_org,pauli_basis_4q) #for the constraints
    
    print(cp.installed_solvers())

    F=cp.norm(variable_rho-targ_mat, tiponorma)  
    
    # Set the remaining constraints
    constraints = [cp.trace(variable_rho) == 1, variable_rho >> 0] 
    
    # Construct the problem.
    objective = cp.Minimize(F)    
    
    ###impose corr constarints
    for i in range(1,256): #not imposing normalization
        if corrs_counts[i-1]=="1":   
            corr_value=cp.real(cp.trace(pauli_basis_4q[i]@org_mat))
            constraints.append(cp.real(cp.trace(variable_rho@pauli_basis_4q[i])) == corr_value)   

    prob = cp.Problem(objective, constraints)
    result = [prob.solve(verbose=True,solver = 'SCS'), variable_rho.value]
    #result_rescaled = np.real(pow(result, 1/n)) # rescale fidelity with the number of parties

    return result

def stato_from_corr(corr_vec, pauli_basis_4q):
    #define pauli matrices for two qubits
    #pauli_basis=[qp.identity(2),qp.sigmax(), qp.sigmay(), qp.sigmaz()]
    #pauli_basis_4q=[]
    #for i1 in range(4):
    #    for i2 in range(4):
    #        for i3 in range(4):
    #            for i4 in range(4):
    #                pauli_basis_4q.append(qp.tensor(pauli_basis[i1], pauli_basis[i2], pauli_basis[i3], pauli_basis[i4]).full())
    
    
    stato=pauli_basis_4q[0]   #lo pongo inizialmente uguale al corr banale
    
    for i in range(1,256): #not including trivial correlator
        stato=stato+pauli_basis_4q[i]*corr_vec[i-1]   #corr_vec[0] corrisponde alla pauli_basis_4q[1] e cosi via 
    stato=1/2**4 *stato
    return stato 

def unnormalize(inp):
    inp=inp.numpy().reshape(256)
    inp = 2*inp -1
    inp= delete(inp,0)
    return inp

def unnormalizeForAttention(inp):
    inp=inp.numpy()
    inp = 2*inp -1
    inp= delete(inp,0)
    return inp

class donatoLoss(nn.Module):
    def __init__(self):
        super(donatoLoss,self).__init__()

    def forward(self, output,target,beta):
        diff = torch.sub(output,target)
        #forcing the purity to one with the regularizer term
        loss = torch.mean(torch.pow(diff,2))+ beta*(torch.sum(torch.pow(output,2)))
        return loss 
    
def fid(a,b):
    r1 = sqrtm(a)
    f = (sqrtm(r1@b@r1))
    return (np.trace(f).real.round(6))**2
    
#note: custom lossAdri works for density matrices
class customLossAdri(nn.Module):

    def __init__(self):
        super(customLossAdri, self).__init__()

        
    def forward(self, output, target):

        tracemy = output.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) 
        diff = torch.sub(output,target)
        loss = torch.mean(torch.pow(diff,2)) - torch.sum((tracemy**2 - tracemy**3 ))
        return loss 
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.0001):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class MSEDiceLoss(nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(MSEDiceLoss, self).__init__()

	def forward(self, inputs, targets, beta=0.1, smooth=0.0001):
		
		#comment out if your model contains a sigmoid or equivalent activation layer
		#inputs = F.sigmoid(inputs)       
		
		#flatten label and prediction tensors
		inputs = inputs.view(-1)
		targets = targets.view(-1)
		
		intersection = (inputs * targets).sum()     
							   
		dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

		diff = torch.sub(inputs,targets)
		mseloss = torch.mean(torch.pow(diff,2))+ beta*(torch.sum(torch.pow(inputs,2)))
		return 1 - 0.6*dice + 0.5*mseloss

class observableBased(nn.Module):
    def __init__(self,M):
        super(observableBased,self).__init__()
        self.M = M
        #self.D = D
        
    def forward(self, inputs, targets):

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        diff = torch.sub(inputs,targets)
        return 0.5*(self.M)*torch.mean(torch.square(torch.abs(diff)))

def dicetest(reconstructions,originals,modelInputs):
    for i in range(10000):
        a = reconstructions[i].view(-1).numpy()
        b = originals[i].view(-1).numpy()
        c = modelInputs[i].view(-1).numpy()

        dif = (dicetest(a,b)- dicetest(b,c))
        if dif>0: print(dif)

    print(np.trace(reconstructions[i].numpy()))

def traceNormTest(m):
    m=m.numpy()
    e = np.linalg.eigvals(m)
    return sum(np.abs(e))


