import torch
import torch.nn as nn
import json
import os
import math
import sys

import util.misc as utils


def train_one_epoch(model, ce_loss, dice_loss, data_loader, optimizer, device):
    
    model.train()
    
    total_ce = 0
    total_dice = 0
    total_loss = 0
    
    n = len(data_loader)
    
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = targets.to(device)
        
        outputs = model(samples)
        loss_ce = ce_loss(outputs, targets)
        loss_dice = dice_loss(outputs, targets)
        loss =  0.5 * loss_ce + 0.5 * loss_dice
        
        total_ce += loss_ce.item()
        total_dice += loss_dice.item()
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    return {
        'loss_ce': total_ce/n,
        'loss_dice': total_dice/n,
        'loss': total_loss/n}
    
    
    
@ torch.no_grad()
def evaluate(model, ce_loss, dice_loss, data_loader, device):
    
    model.eval()
    full_loss = 0
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = targets.to(device)
        
        outputs = model(samples)
        
        loss_ce = ce_loss(outputs, targets.long())
        loss_dice = dice_loss(outputs, targets)
        
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        full_loss += loss.item()
    
    loss_result = full_loss / len(data_loader)
    
    return loss_result
        