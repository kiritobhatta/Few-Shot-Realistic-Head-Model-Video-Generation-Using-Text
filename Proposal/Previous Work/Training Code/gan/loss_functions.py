import torch
import numpy as np
import scipy as sc

EPS = 1e-06

def dis_loss(FDwG1, FDwO1, SDwG2, SDwO2 ):

    return (
        torch.mean(torch.log(EPS + FDwO1)) +
        torch.mean(torch.log(EPS + 1 - FDwG1)) +
        torch.mean(torch.log(EPS + SDwO2)) +
        torch.mean(torch.log(EPS + 1 - SDwG2))
    )

def fdis_loss(FDwG1, FDwO1):
    return (
        torch.mean(torch.log(EPS + FDwO1)) +
        torch.mean(torch.log(EPS + 1 - FDwG1))
        )

def sdis_loss(SDwG2, SDwO2):
    return (
        torch.mean(torch.log(EPS + SDwO2)) +
        torch.mean(torch.log(EPS + 1 - SDwG2))
        )

def gen_loss(FDwG1, SDwG2):
    return (
        torch.mean(torch.log(EPS + FDwG1)) +
        torch.mean(torch.log(EPS + SDwG2))
        )