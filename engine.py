import torch
import torch.nn as nn
import json
import os
import math
import sys

import util.misc as utils


def train_one_epoch(model, ce_loss, dice_loss, data_loader, optimizer, device):
    
    model.train()
    
    for samples, targets in data_loader:
        samples.to(device)
        targets.to(device)
        
        outputs = model(samples)
        loss_ce = ce_loss(outputs, targets)
        loss_dice = dice_loss(outputs, targets)
        loss =  0.5 * loss_ce + 0.5 * loss_dice
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
# @ torch.no_grad()
# def evaluate(model):
    