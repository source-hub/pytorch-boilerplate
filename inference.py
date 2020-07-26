import torch
from config import Config as cfg
from nets.simple_conv import Network
from skimage import io, transform
import argparse
import pickle
def load_model(path=None):
        device=cfg["device"]
        model=Network(10)
        models.load_state_dict(torch.load(path))
        model.eval()
        model.to(device=device)
        return model
@torch.no_grad()
def predict_dataset(test_input):
        net=load_model()
        test_dataset=NewDataset(root=cfg["test_path"])
        test_data=DataLoader(test_dataset,batch_size=4,num_workers=4)
        for _,data in enumerate(train_data,0):
                features,targets=data
                features=features.to(device=device)
                targets=targets.to(device=device)
                outputs=net(features)
                outputs=outputs.to(device=device)
                _, preds = torch.max(outputs,1)
                print()
@torch.no_grad()
def predict_sample(img_path):
        img=io.imread(img_path)
        img=img.to(device=cfg["device"])
        net=load_model(path="./saved_models/new_model.pt")
        with open("./saved_models/new_model.pickle","rb") as f:
                cfg=pickle.load(f)
        idx_to_class=cfg["class_dict"]
        outputs=net(img)
        outputs=outputs.to(device=device)
        _, pred = torch.max(outputs,1)
        print(idx_to_class[pred])

if __name__=="__main__":
        # parser=argparse.ArgumentParser(description="test you network")
        # parser.add_argument("--img",dest="img_path")
        # parser.add_argument("--dir",dest="model_dir")
        predict_sample("./test_images/beaver.jpg")
        

        