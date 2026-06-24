import bagua_model as OldModel
import new_bagua_model as NewModel
import torch
import os
import onnx
from torch.amp import autocast
import sys
import argparse
from onnx import load_model, save_model
from onnxconverter_common.float16 import convert_float_to_float16
# model = OldModel.ResNet(55,15,12,350,148,118)
model = NewModel.ResNet(55,15,4,192,92,82)

parser = argparse.ArgumentParser(description='state_dict to convert to ONNX')
if len(sys.argv) < 3:
    print('Usage: python state_to_onnx.py <model_path> <onnx_path>')
    exit(0)
model_path = sys.argv[1]
new_name = sys.argv[2]
if os.path.exists(model_path):
    print(f'read {model_path}')
    model.load_state_dict(torch.load(model_path, 'cpu'))
else:
    print(f'{model_path} not found')
    exit(0)

dummy_bin = torch.rand((1, 55, 9, 9))
dummy_overall = torch.rand((1, 15))
model.eval()
torch.onnx.export(model, (dummy_bin, dummy_overall), new_name,
        dynamic_axes={'binary': {0: 'batch_size'}, 'overall': {0: 'batch_size'}},
        input_names=['binary', 'overall'],
        output_names=['policy', 'value']
)
onnx.checker.check_model(new_name)
# onnx_model = load_model(new_name)
# trans_model = convert_float_to_float16(onnx_model)
# save_model(trans_model, new_name)
print(f'{new_name} created successfully')