import torch
import os

model_dict = {}

def initiate_load():
    model_path = "data/grid.dat"
    # model_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0))
    model_dict = torch.load(model_path, map_location=torch.device('cpu'))
    return model_dict


