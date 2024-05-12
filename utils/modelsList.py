import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


### MY POSITIONAL ENCODING
def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
	return pos * angle_rates

def positional_encoding(vector):
	positions = len(vector)
	d_model = len(vector)

	#angle_rads = get_angles(np.arange(positions)[:, np.newaxis],np.arange(d_model)[np.newaxis, :],d_model)
	angle_rads = get_angles(np.arange(positions),0,d_model)

	#apply sin to even indices in the array; 2i	
	angle_rads[0::2] = np.sin(angle_rads[ 0::2])
	
	#apply cos to odd indices in the array; 2i+1
	angle_rads[ 1::2] = np.cos(angle_rads[ 1::2])
	
	pos_encoding = angle_rads[...]
	
	return pos_encoding.flatten() + vector

######## DATASETs

class MNIST_AE_Dataset(Dataset):
	def __init__(self, csv_org, csv_noise,transform=None):
		#self.data=pd.read_csv(csv_org) #cancello la colonna di indici        
		#self.data_noise=pd.read_csv(csv_noise) #cancello la colonna di indici
		self.data = csv_org
		self.data_noise = csv_noise
		self.transform=transform #in init ci sono cose che usano le altre funzioni
		
	def __len__(self):        
		return len(self.data)
	
	def __getitem__(self, idx):       
		img=self.data.iloc[idx]
		img=np.array(list([1])+list(img))
		#qui introduce l'input per la cnn, che vuole anche il numero di canali come dimensione (qui 1). In piu'
		# normalizza in 0,1
		img=(np.reshape(img,(16,16,1))+1)/2 
		
		noisy_img=self.data_noise.iloc[idx]
		noisy_img=np.array(list([1])+list(noisy_img))
		noisy_img=(np.reshape(noisy_img,(16,16,1))+1)/2 
		#
		sample=(np.float32(noisy_img), np.float32(img))
		
		#we are not using them
		if self.transform:
			sample=self.transform(sample)
		return sample

class encoderAE(nn.Module):
	def __init__(self,  c2, k, km, pad, strd):

		super(encoderAE, self).__init__()
		self.drop = nn.Dropout(p=0.2)
		# encoder
		self.enc1 = nn.Conv2d(1, c2,kernel_size = (k,k),padding=pad, stride = strd)
		self.max = nn.MaxPool2d(kernel_size=(km,km))

		self.gelu = nn.GELU()
		self.relu = nn.ReLU()
		
	def forward(self,x):

		x = self.gelu(self.enc1(x))
		x = self.max(x)

		return x
  
class decoderAE(nn.Module):

	def __init__(self,c2, k, strd):
		super(decoderAE,self).__init__()

		self.enct1 = nn.ConvTranspose2d(c2,1, kernel_size = (k,k), stride = strd)
		self.T = nn.Tanh()
		self.relu = nn.ReLU()

	def forward(self,x):
		x = self.relu(self.enct1(x))
		return x


class attentionDataset(Dataset):
	def __init__(self, csv_org, csv_noise,transform=None):
		self.data = csv_org
		self.data_noise = csv_noise
		self.transform=transform #in init ci sono cose che usano le altre funzioni
		
	def __len__(self):        
		return len(self.data)
	
	def __getitem__(self, idx):       
		img=self.data.iloc[idx]
		img=np.array(list([1])+list(img))
		#qui introduce l'input per la cnn, che vuole anche il numero di canali come dimensione (qui 
		
		# normalizza in 0,1
		img = (np.array(img)+1)/2 
		
		noisy_img=self.data_noise.iloc[idx]
		
		noisy_img=np.array(list([1])+list(noisy_img))
		noisy_img = (np.array(noisy_img)+1)/2
		sample=(np.float32(noisy_img), np.float32(img))

		return sample



class PositionalAttentionDataset(Dataset):
	def __init__(self, csv_org, csv_noise,transform=None):
		self.data = csv_org
		self.data_noise = csv_noise
		self.transform=transform #in init ci sono cose che usano le altre funzioni
		
	def __len__(self):        
		return len(self.data)
	
	def __getitem__(self, idx):       
		img=self.data.iloc[idx]
		img=np.array(list([1])+list(img))
		#qui introduce l'input per la cnn, che vuole anche il numero di canali come dimensione (qui 
		
		# normalizza in 0,1
		img = (np.array(img)+1)/2 
		
		noisy_img=self.data_noise.iloc[idx]
		
		noisy_img=np.array(list([1])+list(noisy_img))
		noisy_img = (np.array(noisy_img)+1)/2
		noisy_img = positional_encoding(noisy_img)
		sample=(np.float32(noisy_img), np.float32(img))

		return sample



class NormalizedAttentionDataset(Dataset):
	#here we normalize the data inside 0,1
	def __init__(self, csv_org, csv_noise,transform=None):
		self.data = csv_org
		self.data_noise = csv_noise
		self.transform=transform #in init ci sono cose che usano le altre funzioni
		
	def __len__(self):        
		return len(self.data)
	
	def __getitem__(self, idx):       
		img=self.data.iloc[idx]
		img=np.array(list([1])+list(img))
		#qui introduce l'input per la cnn, che vuole anche il numero di canali come dimensione (qui 
		
		# normalizza in 0,1
		img = (np.array(img)+1)/2
		img = img/sum(img)
		
		noisy_img=self.data_noise.iloc[idx]
		
		noisy_img=np.array(list([1])+list(noisy_img))
		noisy_img = (np.array(noisy_img)+1)/2
		noisy_img = noisy_img/sum(noisy_img)
		sample=(np.float32(noisy_img), np.float32(img))

		return sample

class Netbeta(nn.Module):
	def __init__(self, input_len, out_channel, nheads, dim_ff ):
		super().__init__()

		kernel = 3
		kernel2 = 3
		

		padd = 1
		strid1 = 1
		strid2 = 1
		model_dim = int((input_len - kernel + 2*padd)/strid1)+1


		#ENCODING
		self.conv1 = nn.Conv1d(1, out_channel, kernel_size = kernel, padding = padd,stride = strid1 )
		self.conv2 = nn.Conv1d(out_channel, 1, kernel_size = kernel2, padding = padd, stride = strid2 )
		

		#RECONSTRUCTION

		self.enc_transf = nn.TransformerEncoderLayer(d_model = model_dim, nhead = nheads, dim_feedforward = dim_ff, batch_first = True, norm_first=True)
		self.relu = torch.nn.ReLU()
		self.enc_stack = nn.TransformerEncoder(self.enc_transf, num_layers=1)
		self.T = torch.nn.Tanh()
		self.G = torch.nn.GELU()


	def forward(self, x):

		#ENCODING

		x = self.G(self.conv1(x)) #this for pauli's
		#x= F.selu(self.conv1(x))

		#DECODING

		x= self.G(self.enc_stack(x))

		x = self.relu(self.conv2(x))
		x = torch.flatten(x,1) 

		return x

class FullTransformer(nn.Module):
	def __init__(self, input_len, out_channel, nheads, dim_ff ):
		super().__init__()

		kernel = 3
		kernel2 = 3
		

		padd = 1
		strid1 = 1
		strid2 = 1
		model_dim = int((input_len - kernel + 2*padd)/strid1)+1


		#ENCODING
		self.conv1 = nn.Conv1d(1, out_channel, kernel_size = kernel, padding = padd,stride = strid1 )
		self.conv1b = nn.Conv1d(1, out_channel, kernel_size = kernel, padding = padd,stride = strid1 )
		self.conv2 = nn.Conv1d(out_channel, 1, kernel_size = kernel2, padding = padd, stride = strid2 )
		

		#RECONSTRUCTION

		self.transf = nn.Transformer(d_model = model_dim, nhead = nheads, dim_feedforward = dim_ff, batch_first = True, norm_first=True, num_encoder_layers = 1, num_decoder_layers = 1)
		self.relu = torch.nn.ReLU()
		self.T = torch.nn.Tanh()
		self.G = torch.nn.GELU()


	def forward(self, x,tgt):

		#ENCODING

		x = self.G(self.conv1(x)) #this for pauli's
		#x= F.selu(self.conv1(x))
		

		#DECODING
		y = self.G(self.conv1b(tgt))
		x= self.G(self.transf(x,y))

		x = self.relu(self.conv2(x))
		x = torch.flatten(x,1) 

		return x

