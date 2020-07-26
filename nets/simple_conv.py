import torch
from torch import nn
import torch.nn.functional as f

class Network(nn.Module):
	def __init__(self,num_classes):
			super(Network,self).__init__()
			self.conv1=nn.Conv2d(3,16,5)#200->98
			self.conv2=nn.Conv2d(16,32,5)#98->47
			self.conv3=nn.Conv2d(32,64,6)#47->21
			self.conv4=nn.Conv2d(64,128,6)#21->8
			self.conv5=nn.Conv2d(128,256,3)#8->1
			self.pool=nn.MaxPool2d(2,2)
			self.fc1=nn.Linear(256*3*3,512)
			self.fc2=nn.Linear(512,256)
			self.fc3=nn.Linear(256,num_classes)
			self.drp=nn.Dropout(p=0.4)
	def forward(self,x):
			x=self.pool(f.relu(self.conv1(x)))
			x=self.pool(f.relu(self.conv2(x)))
			x=self.pool(f.relu(self.conv3(x)))
			x=self.pool(f.relu(self.conv4(x)))
			x=self.pool(f.relu(self.conv5(x)))
			x=x.view(-1,256*3*3)
			x=f.relu(self.fc1(x))
			x=self.drp(x)
			x=f.relu(self.fc2(x))
			x=self.drp(x)
			x=self.fc3(x)
			return x