import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset

from tqdm import tqdm
import torch.nn as nn
import random
import torch.nn.functional as F

import matplotlib.patches as mpatches
from models.modelsList import *

from qutip import fidelity, Qobj
from models.utils import *


foldertype= 'A/'
url ='data/fromMeasurementSettings/'+foldertype
testorg = pd.read_csv(url+'test_org.csv')
testnoise = pd.read_csv(url+'test_noise.csv')

batch_size= 200

fidNoisyTarget = []
fidProjectedTarget =[]
PurNoisyTarget = []
PurProjectedTarget =[]

pauliBasis = generatePaulimatrices()

tiponorma = 'fro'
def unnormalizeForAttention(inp):
    inp=inp.numpy()
    inp = 2*inp -1
    inp= delete(inp,0)
    return inp


def globalDepo(rm):
	p= 0.
	i = np.eye(16,16)
	dep = (1-p)*rm + (p/16.)*i
	return dep
pbasis = generatePaulimatrices()

def counting(test_dataload):
	for el in test_dataload:
		noisy_img, org_img= el
		print(noisy_img.shape)
		trues = np.round(noisy_img[3],8)== np.round(org_img[3],8)
		tot = [1 for el in trues if el==True]
		print(len(tot)-1) #lent (tot) -1 perche' c'e' anche la identita'
		break


def cycle(batc, originals):
    batc = batc.squeeze().cpu()
    originals = originals.squeeze().cpu()
    pb = generatePaulimatrices()
    f = open('data/fromMeasurementSettings/'+foldertype+'/corrsCountList.txt', 'r')
    corrs_counts = f.read()
    print(len(corrs_counts))
    tiponorma = "fro"

    manySteps = []
    for el,orig in zip(batc,originals):
        el_vec = unnormalizeForAttention(el)
        orig_vec = unnormalizeForAttention(orig)
        corrttproj_dm=proj(corrs_counts, el_vec, orig_vec,tiponorma,pb)[1]
        newEl = [(np.trace(corrttproj_dm @ b)+1)/2 for b in pb]
        manySteps.append(newEl)
    return manySteps

#first off, generate the basis once!

	
import argparse
if __name__=="__main__":

	#init = 200
	#fin = 300
	parser = argparse.ArgumentParser()
	parser.add_argument('--init',type = int ,help='el init')
	parser.add_argument('--fin',type = int ,help='la fin')
	parser.add_argument('--filenum',type=int)

	args = parser.parse_args()
	
	print('FIN',args.fin)
	testset=attentionDataset(testorg[args.init:args.fin], testnoise[args.init:args.fin])

	del(testorg)
	del(testnoise)

	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = 'cpu'
	print(device)
	model = Netbeta(256,3,1,256)

	test_dataloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size,
    shuffle=False,
    pin_memory=True)

	#TESTING THE NUMBER OF CORRELATORS
	counting(test_dataloader) 

	#UPLOAD MODEL
	model = torch.load('models/measurementSettingsModels/Dep0)TransformerDUEMeasurementSettingsA-MSE.pth',map_location=torch.device('cpu'))

	
	reconstructions = []
	originals = []
	modelInputs = []

	with torch.no_grad():
		for inputs in test_dataloader:
			model.eval()

			noisy_img, org_img= inputs
			noisy_img = noisy_img.to(device)
			org_img = org_img.to(device)
			noisy_img = torch.reshape(noisy_img,(noisy_img.shape[0],1,256))
			denoised_img=model(noisy_img)
			denoised_img = cycle(denoised_img, noisy_img)

		for _ in range(3):
			denoised_img = torch.Tensor(denoised_img).to(device)
			#denoised_img = model(torch.reshape(denoised_img,(args.fin-args.init,1,256)))
			denoised_img = model(torch.reshape(denoised_img,(denoised_img.shape[0],1,256)))
			denoised_img = cycle(denoised_img, noisy_img)
        
		denoised_img = torch.Tensor(denoised_img).to(device)

		reconstructions.extend(denoised_img.cpu().detach())
		originals.extend(org_img.cpu().detach())
		modelInputs.extend(noisy_img.cpu().detach())

	#FINAL PostPROCESSING, I SQUASH EVERYTHING IN THE FEASIBLE SET and reconstruct the correct outputs
	f = open('data/fromMeasurementSettings/'+foldertype+'/corrsCountList.txt', 'r')
	corrs_counts = f.read()
	for corretta, rumorosa, originale in zip(reconstructions, modelInputs, originals):

		corretta_vec = unnormalizeForAttention(corretta)
		rumorosa_vec = unnormalizeForAttention(rumorosa)
		originale_vec = unnormalizeForAttention(originale)

		corrttproj_dm=proj(corrs_counts, corretta_vec, rumorosa_vec,tiponorma,pauliBasis)[1]
		org_dm=stato_from_corr(originale_vec, pauliBasis)
		org_dmB = globalDepo(org_dm)
		print('org fidelity',fidelity(Qobj(org_dmB),Qobj(org_dm)))
		noisy_dm=stato_from_corr(rumorosa_vec, pauliBasis)

		#print('PROJ PURiTY', np.trace(corrttproj_dm@corrttproj_dm)) 
    
		fidNoisyTarget.append(fidelity(Qobj(noisy_dm), Qobj(org_dmB)))
		fidProjectedTarget.append(fidelity(Qobj(corrttproj_dm), Qobj(org_dmB)))

		PurNoisyTarget.append((Qobj(noisy_dm)*Qobj(noisy_dm)).tr())
		PurProjectedTarget.append((Qobj(corrttproj_dm)*Qobj(corrttproj_dm)).tr() )

	print(np.mean(fidNoisyTarget))
	print(np.mean(fidProjectedTarget))

	np.save('FPout/p0/fidNoisyTarget'+str(args.filenum)+'.npy',fidNoisyTarget )
	np.save('FPout/p0/fidProjectedTarget'+str(args.filenum)+'.npy',fidProjectedTarget)

	np.save('FPout/p0/PurNoisyTarget'+str(args.filenum)+'.npy',PurNoisyTarget)
	np.save('FPout/p0/PurProjectedTarget'+str(args.filenum)+'.npy',PurProjectedTarget)




	