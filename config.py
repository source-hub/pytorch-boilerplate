import torch
from torchvision import transforms
# Config={
#     "device":torch.device("cuda:0"),
#     "model_save_path":"saved_models/",
#     "train_path":"data/train",
#     "test_path":"data/test"
# }
# if not torch.cuda.is_available():
#         Config["device"]=torch.device("cpu:0")
    
# train_transforms=transforms.Compose([
# 		        transforms.RandomResizedCrop(200),
#         	    transforms.RandomHorizontalFlip(),
#         	    transforms.ToTensor(),
#         	    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#     		    ])
# inference_transforms=transforms.Compose([
#                 transforms.RandomResizedCrop(200),
#                 transforms.ToTensor()
# ])
# Config["train_transforms"]=train_transforms
# Config["inference_transforms"]=inference_transforms

#config was a dictionary before and now since as its state has to be maintained, it has been converted to a class instead 
class Config(dict):
    def __init__(self):
            self.device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
            self.model_save_path="saved_models/"
            self.train_path="data/train"
            self.test_path="data/test"
            self.train_transforms=transforms.Compose([
		        transforms.RandomResizedCrop(200),
        	    transforms.RandomHorizontalFlip(),
        	    transforms.ToTensor(),
        	    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    		    ])
            self.inference_transforms=transforms.Compose([
                transforms.RandomResizedCrop(200),
                transforms.ToTensor()
                ])
    def __dir__(self):return self.__dict__
    def __getitem__(self,attr):
            return self.__dict__[attr]
    def __setattr__(self,attr,val):
            self.__dict__[attr]=val
    def __repr__(self):
            return repr(self.__dict__) 
    __setitem__=__setattr__
    __getattr__=__getitem__
    def __getstate__(self):
        """
        #refer:https://stackoverflow.com/questions/12101574/why-does-pickle-dumps-call-getattr
        """          
        return self.__dict__