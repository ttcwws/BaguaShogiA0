from bagua_model import ResNet
import torch
import torch.nn.functional as F
from file import *
from random import randint
import numpy as np
import onnx
import os
import time
from buffer import Buffer
batch_size = 512
value_ratio = 1
device = 'cuda:0'
def ind_init():
    S_KING = 1
    S_GOLD = 2
    S_SILVER = 3
    S_KNIGHT = 4
    S_LANCE = 5
    S_ROOK = 6
    S_BISHOP = 7
    S_PAWN = 8
    S_HORIZONTAL = 0x10
    S_PROMOTED = 0x20
    S_SIDE = 0x40
    ind = [0] * 55
    # ind[0] = 0x7F
    for i in range(S_KING, S_PAWN + 1):
        ind[i] = i
        ind[i+27] = i | S_SIDE
    for i in range(S_GOLD, S_PAWN + 1):
        ind[i+13] = i | S_HORIZONTAL
        ind[i+13+27] = i | S_HORIZONTAL | S_SIDE
    for i in range(S_SILVER, S_PAWN + 1):
        ind[i+6] = i | S_PROMOTED
        ind[i+6+27] = i | S_PROMOTED | S_SIDE
        ind[i+6+13] = i | S_PROMOTED | S_HORIZONTAL
        ind[i+6+13+27] = i | S_PROMOTED | S_HORIZONTAL | S_SIDE
    ind = torch.tensor(ind, dtype=torch.int64)
    ind = ind.reshape(-1,1,1).expand(-1,9,9)
    return ind
ind = ind_init()
def to_bin(color):
    global ind
    a = color.unsqueeze(1) == ind.unsqueeze(0)
    a[:,0] = color != 0x7F
    return a

CHECK_INTERVAL = 121
def train(model:torch.nn.Module, optim,train_buffer:Buffer,device, path_prefix:str):
    step = 0
    lp, lv = 0, 0
    # idx = torch.randperm(len(train_buffer))
    for step in range(1, len(train_buffer) // batch_size + 1):
        model.train()

        s = time.time()
        color, input_overall, policy_targets, value_targets = train_buffer.choice(batch_size)
        # color, input_overall, policy_targets, value_targets = train_buffer[idx[(step-1)*batch_size:step*batch_size]]
        input_bin = to_bin(color)
        input_bin, input_overall, policy_targets, value_targets = input_bin.to(torch.float32), input_overall.to(torch.float32), policy_targets.to(torch.float32), value_targets.to(torch.float32)
        data_bin, data_overall, data_policy, data_value = input_bin.to(device), input_overall.to(device), policy_targets.to(device), value_targets.to(device).unsqueeze(-1)
        pred_policy, pred_value = model(data_bin, data_overall)
        policy_loss = F.cross_entropy(pred_policy, data_policy)
        value_loss = F.mse_loss(pred_value, data_value)
        loss = policy_loss + value_loss * value_ratio
        lp += policy_loss.item() * batch_size
        lv += value_loss.item() * batch_size
            # loss = value_loss
        optim.zero_grad()
        loss.backward()
            # print(loss)
        optim.step()
            # if 0 == randint(0,10):
            #     with torch.no_grad():
            #         _, np = model(data_bin, data_overall)
            #         print(data_value[0].item(), pred_value[0].item(), np[0].item())
        # verification
        l = CHECK_INTERVAL * batch_size
        if step % CHECK_INTERVAL == 0:
            print(f'step {step}, train loss: {(lp + lv) / l}, policy loss: {lp / l}, value loss: {lv / l}')
            if lv / l <= 0.20: break
            lp = lv = 0
        # lossp, lossv = 0, 0
        # model.eval()
        # with torch.no_grad():
        #     for i in range(0, len(valid_buffer), batch_size):
        #         r = min(i+batch_size, len(valid_buffer))
        #         color, overall, policy, value = valid_buffer[i:r]
        #         bin = to_bin(color)
        #         bin, overall, policy, value = bin.to(torch.float32), overall.to(torch.float32), policy.to(torch.float32), value.to(torch.float32)
        #         pred_policy, pred_value = model(bin.to(device), overall.to(device))
        #         policy_loss = F.cross_entropy(pred_policy, policy.to(device), reduction='sum')
        #         value_loss = F.mse_loss(pred_value, value.unsqueeze(-1).to(device), reduction='sum')
        #         lossp += policy_loss.item()
        #         lossv += value_loss.item()
                    # os.system('pause')
                             
        # test_loss = (lossp + lossv * value_ratio) / len(valid_buffer)
        # print(f'test: avg total loss: {test_loss}, avg policy loss: {lossp / len(valid_buffer)}, avg value loss: {lossv / len(valid_buffer)}')
        # print(f'train: avg total loss: {train_loss}, avg policy loss: {lp / l}, avg value loss: {lv / l}')
        # print(f'test loss - train loss: {test_loss - train_loss}')
        # if test_loss > last_valid_loss:
        #     count += 1
        # else: count = 0
        # if count >= 2:
        #     break
        # last_valid_loss = test_loss
        # print()
    torch.save(model.state_dict(), path_prefix + 'latest_state.pt')
    torch.save(optim.state_dict(), path_prefix + 'latest_optim.pt')
            # 保存ONNX模型
    dummy_bin = torch.rand((1, 55, 9, 9), device=device)
    dummy_overall = torch.rand((1, 15), device=device)
    model.eval()
    torch.onnx.export(model, (dummy_bin, dummy_overall), path_prefix + 'latest.onnx',
            dynamic_axes={'binary': {0: 'batch_size'}, 'overall': {0: 'batch_size'}},
            input_names=['binary', 'overall'],
            output_names=['policy', 'value']
            )
    onnx.checker.check_model(path_prefix + 'latest.onnx')
    print(f'{time.time()-s}s')

