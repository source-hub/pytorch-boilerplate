import torch
import time
import os
import datetime
from torchvision import transforms,datasets
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from helpers.utils import *
from config import Config
from nets.simple_conv import Network 
from torch import nn
import pickle
def main(epochs=20,dest_name=None):
    cfg=Config()
    device=cfg["device"]
    train_dataset=NewDataset(root=cfg["train_path"])
    train_data=DataLoader(train_dataset,batch_size=6,num_workers=4)
    net=Network(10)
    net.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    since = time.time()
    print(train_dataset.class_to_idx)
    print("Training the model using :{}".format(device))
    print("Epochs:{}".format(epochs))
    cfg["Epochs"]=epochs
    cfg["class_dict"]={train_dataset.class_to_idx[k]:k for k in train_dataset.class_to_idx}
    for i in range(epochs):
        phase="train"
        running_loss=0.0
        running_corrects=0
        for j,data in enumerate(train_data,0):
                features,targets=data
                features=features.to(device=device)
                targets=targets.to(device=device)
                optimizer.zero_grad()
                outputs=net(features)
                with torch.set_grad_enabled(phase=="train"):
                    outputs=outputs.to(device=device)
                    _, preds = torch.max(outputs,1)
                    loss=criterion(outputs,targets)
                    if phase=="train":
                        loss.backward()
                        optimizer.step()
                    running_loss+=loss.item()*features.size(0)
                    running_corrects+=torch.sum(preds==targets.data)
        epoch_loss = running_loss / len(train_data)
        epoch_acc = running_corrects.double() / len(train_data)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
        time_elapsed = time.time() - since
    format_str='Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    print(format_str)
    cfg["accuracy"]=epoch_acc
    cfg["training_time"]=format_str
    _save(net,dest_name=dest_name,cfg=cfg)
def _save(model,dest_name=None,cfg=None):
    if not dest_name:
        dest_name=str(datetime.datetime.now())
    model_file_name=dest_name+".pt"
    new_dir=os.path.join(cfg["model_save_path"],dest_name)
    os.mkdir(new_dir)
    model_cfg_path=os.path.join(new_dir,dest_name+".pickle")
    with open(model_cfg_path,"wb") as f:
        pickle.dump(cfg,f)
    torch.save(model.state_dict(),os.path.join(new_dir,model_file_name))
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--save",type=str,dest="save_dir")
    args=parser.parse_args()
    if not args.save_dir:
        raise ValueError("Please provide the dir name for the model's metadata to be saved. This can be done by adding the flag --save <dir name>")
    main(20,args.save_dir)


    
