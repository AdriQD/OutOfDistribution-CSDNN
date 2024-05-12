import numpy as np
import cvxpy as cp
import qutip as qp
import pandas as pd
from cvxpy.atoms.norm_nuc import normNuc
from multiprocessing import Pool
import os
import gc


#neeeded functions

def ternary (n):

	if n == 0:
		return [0,0,0,0]
	nums = []
	while n:
		n, r = divmod(n, 3)
		nums.append(str(r))
	ter=''.join(reversed(nums))
	list_ter=[int(x) for x in ter[0:]]  
	size=len(list_ter)
	moresize=[0]*(4-size)
	return moresize+list_ter  
	 


##############################computing pauli number list
def paulinumber (stringa):
	array = [0] * 4
	for i in range(0,4):  #it does not arrive to 4
		if stringa[i] == 'X':
			array[i]=1
		if stringa[i] == 'Y':
			array[i]=2
		if stringa[i] == 'Z':
			array[i]=3            
	return array


#####from pauli number list to pauli number in base 3, it associates a number from 0 to 80
def paulinumberB3 (pauli):  
	return int((pauli[0]-1) * 3**3 + (pauli[1]-1) * 3**2 + (pauli[2]-1) * 3**1 + (pauli[3]-1) * 3**0)

######we need also a numbering for the correlators, this time I need base 4.
def momnumberB4 (i1,i2,i3,i4):  ###
	return int(i1* 4**3 + i2* 4**2 + i3* 4**1 + i4* 4**0)

#print(momnumberB4(3,3,3,3))

##########given a certain correlator which are the rows that are relevant? this function returns the list of rows
def whichrows(i1,i2,i3,i4):
	i1=i1-1; i2=i2-1; i3= i3-1; i4=i4-1  ##now zero, namely the identity, corresponds to -1. 
	rows=[]
	for i in range (0,81):
		ter=ternary(i)
		if i1==-1 or i1==ter[0]: 
			if i2==-1 or i2==ter[1]: 
				if i3==-1 or i3==ter[2]: 
					if i4==-1 or i4==ter[3]:
						rows=rows+[i]
	return rows

  

#######################input a number (base 10) returns the list in base 4
def quaternary (n):
	if n > 255:
		return "error: n must be < 255"
	if n == 0:
		return [0,0,0,0]
	nums = []
	while n:
		n, r = divmod(n, 4)
		nums.append(str(r))
	ter=''.join(reversed(nums))
	list_ter=[int(x) for x in ter[0:]]  
	size=len(list_ter)
	moresize=[0]*(4-size)
	return moresize+list_ter





def intersection(lst1, lst2):
	lst3 = [value for value in lst1 if value in lst2]
	return lst3


def how_many_corr(r):
	vec=[]
	for i in range(len(r)):
		vec.append(float(r[i]))
	a=str("number of corr used: {}".format(sum(vec)))
	return a


###dati alcuni meas settings ritorna la lista di 0 o 1 dei correlatori implicati.

def corrs_from_setts(chosen_setts):
	corr_string=""
	#loop su tutti i correlatori (in base 10)
	for i_corr in range(1, 256): #i don't count the trivial corr, goes from 1 to 255
		corr_vec=quaternary(i_corr)
		#settings compatibili
		comp_setts=whichrows(corr_vec[0],corr_vec[1],corr_vec[2],corr_vec[3])
		inter=intersection(comp_setts, chosen_setts)
		#print(inter)
		if len(inter)>0:
			#print("yes")
			corr_string=corr_string+"1"
		else:
			#print("no")
			corr_string=corr_string+"0"
	return corr_string



#voglio fare che corrs_counts sia una stringa di 0 e 1, che mi dice solo se considerare o meno il correlatore.
def pauli_basis4q():

	pauli_basis=[qp.identity(2),qp.sigmax(), qp.sigmay(), qp.sigmaz()]
	pauli_basis_4q=[]
	for i1 in range(4):
		for i2 in range(4):
			for i3 in range(4):
				for i4 in range(4):
					pauli_basis_4q.append(qp.tensor(pauli_basis[i1], pauli_basis[i2], pauli_basis[i3], pauli_basis[i4]).full())
	return pauli_basis_4q


def minF(opt,corrs_counts, target, Pbasis):    
	#define pauli matrices for two qubits

	pauli_basis_4q = Pbasis
	variable_rho = cp.Variable((16,16), hermitian=True) #two qubit dm
	
	targ_mat=(target*target.dag()).full()
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
	result = [prob.solve(verbose=False, solver='MOSEK'), variable_rho.value]
	#result_rescaled = np.real(pow(result, 1/n)) # rescale fidelity with the number of parties

	return result

def DepoMinF(opt,corrs_counts, target, Pbasis):    
	#define pauli matrices for two qubits
	
	p= 0.1
	print('depo val',p)
	i = np.eye(16,16)
	#dep = (1-p)*rm + (p/16.)*i

	pauli_basis_4q = Pbasis
	variable_rho = cp.Variable((16,16), hermitian=True) #two qubit dm
	
	targ_mat=(target*target.dag()).full()
	depotarget = (1-p)*targ_mat + (p/16.)*i
	
	#print('Fid target original', qt.fidelity(Qobj(depotarget),Qobj(target_mat)))
	F=0  #se non faccio niente mi ritorna una rappresentazione cs
	
	# Set the remaining constraints
	constraints = [cp.trace(variable_rho) == 1, variable_rho >> 0] 
	
	if opt=="minF":
		F = cp.real(cp.trace(variable_rho@depotarget))     
	if opt=="cs":
		F= normNuc(variable_rho)
		# Set the remaining constraints
		constraints = [cp.trace(variable_rho) == 1] #in questo caso non uso positivity 

	# Construct the problem.
	objective = cp.Minimize(F)
	
	###impose corr constarints
	for i in range(1,256): #not imposing normalization
		if corrs_counts[i-1]=="1":   
			corr_value=cp.real(cp.trace(pauli_basis_4q[i]@depotarget))
			constraints.append(cp.real(cp.trace(variable_rho@pauli_basis_4q[i])) == corr_value)   

	prob = cp.Problem(objective, constraints)
	result = [prob.solve(verbose=False, solver='MOSEK'), variable_rho.value]
	#result_rescaled = np.real(pow(result, 1/n)) # rescale fidelity with the number of parties

	return result

def export_corrs(stato, Pbasis):
	#define pauli matrices for two qubits
	pauli_basis_4q= Pbasis

	corr_vec=[]
	for i in range(1,256): #not including trivial correlator
		corr_value=(np.trace(pauli_basis_4q[i]@stato.full()))
		corr_vec.append(corr_value)
	return corr_vec


def stato_from_corr(corr_vec,Pbasis):
	#define pauli matrices for two qubits
	pauli_basis_4q= Pbasis

	stato=pauli_basis_4q[0]   #lo pongo inizialmente uguale al corr banale
	
	for i in range(1,256): #not including trivial correlator
		stato=stato+pauli_basis_4q[i]*corr_vec[i-1]   #corr_vec[0] corrisponde alla pauli_basis_4q[1] e cosi via 
	stato=1/2**4 *stato
	return stato      

	
def Generation( phonynumber):
	pb = pauli_basis4q()
	
	if phonynumber%1000 == False: print(phonynumber)

	target=qp.rand_ket_haar(16)
	target_dm=target*target.dag()
	#stato vero   
	df = np.real(export_corrs(target_dm,pb))

	#stato CS
	cs_rho=qp.Qobj(np.array(minF("cs",corrs_counts, target, pb)[1]))
	
	df_cs = np.real(export_corrs(cs_rho,pb))
	
	return [df,df_cs]
	

def GenerationDepo( phonynumber):

	pb = pauli_basis4q()
	if phonynumber%1000==False: print(phonynumber)

	target=qp.rand_ket_haar(16)
	target_dm=target*target.dag()
	#stato vero   
	p=0.1
	i = np.eye(16,16)
	depotarget = (1-p)*target_dm + (p/16.)*i
	
	df = np.real(export_corrs(depotarget,pb))
	#stato CS	
	cs_rho=qp.Qobj(np.array(DepoMinF("cs",corrs_counts, target, pb)[1]))
	df_cs = np.real(export_corrs(cs_rho,pb))
	
	return [df,df_cs]



def generateRandHaars(howmany):
	s = []
	
	for _ in range(howmany):
		target=qp.rand_ket_haar(16)
		target_dm=target*target.dag()
		s.append(target_dm)
	return s



if __name__=='__main__':

	#print('dephasing value:', 0.1)
	gc.enable()
	#scegliamo i meas setts <----------
	chosen_setts=[0,40,80] #XXXX, YYYY, ZZZZ
	#chosen_setts=[0] 
	#chosen_setts=[0,1]
	#chosen_setts=[0,1,2]
	#chosen_setts=[0,1,2,3]
	#chosen_setts=[0,1,2,3,4]
	#chosen_setts=[0,1,2,3,4,5]
	#chosen_setts=[0,1,2,3,4,5,6]
	#chosen_setts=[0,1,2,3,4,5,6,7]
	#chosen_setts=[0,1,2,3,4,5,6,7,8]  ###----> 
	
	###usiamo funzione per trovare stringa che descrive i correlatori implicati
	corrs_counts=corrs_from_setts(chosen_setts)

	Mtrain = 8
	Mtest = 2

	num_process_available = len(os.sched_getaffinity(0))
	print("processes available:",num_process_available)

	#trainDM = generateRandHaars(Mtrain)
	phonytrain = [el for el in range(Mtrain)]
	phonytest = [el for el in range(Mtest)]


	with Pool(2) as pone:
		print('num process in train:', pone)
		trainData = list(map(GenerationDepo,phonytrain))


	pbb = pauli_basis4q()

	for el in np.array(trainData)[:,0]:
		#ss = stato_from_corr(el,pbb)
		#print(np.trace(ss))
		print(el[-1])

	for el in np.array(trainData)[:,1]:
		print("CS RECONSTRUCTIONS")
		#ss = stato_from_corr(el,pbb)
		#print(np.trace(ss))
		print(el[-1])
		
	df = pd.DataFrame(data = np.array(trainData)[:,0])
	df_cs = pd.DataFrame(data = np.array(trainData)[:,1])

	#df.to_csv('depo02Ctrain_org.csv', index=False)
	#df_cs.to_csv('depo02Ctrain_noise.csv', index=False)

	print('saved train sets with shape',df.shape,df_cs.shape)

	del(df)
	del(df_cs)
	del(trainData)
	del(phonytrain)


	with Pool(num_process_available-5) as p:
		testData = list(map(GenerationDepo,phonytest ))

	df = pd.DataFrame(data = np.array(testData)[:,0])
	df_cs = pd.DataFrame(data = np.array(testData)[:,1])

	#df.to_csv('depo0025test_org.csv', index=False)
	#df_cs.to_csv('depo0025test_noise.csv', index=False)
	print('saved test sets with shape',df.shape,df_cs.shape)

	del(df)
	del(df_cs)
	del(testData)
	del(phonytest)

	
	print("len(corrs_counts) is:",len(corrs_counts))

	print("how_many_corr(corrs_counts) is:", how_many_corr(corrs_counts))


	with open('corrsCountList.txt', 'w') as a:
		a.write(corrs_counts)
		a.close()

	collected = gc.collect()
	print("Garbage collector: collected","%d objects." % collected)