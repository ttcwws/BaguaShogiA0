from bagua_model import *
from trainer import *
import numpy as np
import os
import shutil
import zlib
import onnx
from torch.amp import autocast
MAX_BUFFER_SIZE = 899091
VALID_RATIO = 0.1
VALID_BUFFER_SIZE = int(MAX_BUFFER_SIZE * VALID_RATIO / (1 - VALID_RATIO))
# for i in range(100):
#     print(np.random.dirichlet([0.03,0.03,0.03]))
device = torch.device('cuda:0')
lr = 0.006
model = ResNet(55,15,4,280,112,98).to(device)
dummy_bin = torch.rand((1, 55, 9, 9), dtype=torch.half)
dummy_overall = torch.rand((1, 15), dtype=torch.half)
model.eval()
# torch.onnx.export(model, torch.rand((1,57,9,9)), 'latest.onnx',opset_version=11)
# exit(0)
# sm = torch.jit.script(model)
# sm.save('../latest.pt')
# exit(0)
# ml = torch.jit.load('../latest.pt',device)
# ml.eval()
# ml.load_state_dict(model.state_dict())
# ml=torch.jit.script(ml)
# ml.save('../latest.pt')
# exit(0)

opt = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=2e-4, momentum=0.9)
if os.path.exists('../latest_state.pt'):
    print('read latest.pt')
    model.load_state_dict(torch.load('../latest_state.pt',device))
    if os.path.exists('../latest_optim.pt'):
        opt.load_state_dict(torch.load('../latest_optim.pt',device))
        opt.param_groups[0]['lr'] = lr
        opt.param_groups[0]['weight_decay'] = 2e-4
    # dummy_bin = torch.rand((1, 57, 9, 9), device=device)
    # dummy_overall = torch.rand((1, 15), device=device)
    # model.eval()
    # torch.onnx.export(model, (dummy_bin, dummy_overall), '../latest.onnx',
    #         dynamic_axes={'binary': {0: 'batch_size'}, 'overall': {0: 'batch_size'}},
    #         input_names=['binary', 'overall'],
    #         output_names=['policy', 'value']
    #         )
    # onnx.checker.check_model('../latest.onnx')

else: print('using new model')
# dummy_bin = torch.rand((1, 55, 9, 9), device=device)
# dummy_overall = torch.rand((1, 15), device=device)
# model.eval()
# torch.onnx.export(model, (dummy_bin, dummy_overall), '../latest1.onnx',
#             dynamic_axes={'binary': {0: 'batch_size'}, 'overall': {0: 'batch_size'}},
#             input_names=['binary', 'overall'],
#             output_names=['policy', 'value']
#             )
# exit(0)
# for p in model.named_parameters():
#     print(p)
# exit(0)
# print number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')
if os.path.exists('checkpoint'):
    f = open('checkpoint', 'r')
    s = int(f.read())
    f.close()
else: s = 0
checkpoint = 0
szs = [s]

train_buffer = Buffer(MAX_BUFFER_SIZE)
train_buffer.append_folder('./train_buffer/')
print('train buffer size:', len(train_buffer))
# valid_buffer = Buffer('./valid_buffer/', VALID_BUFFER_SIZE)
# print(np.load(f'../value_target{checkpoint}.npz'))

def load_zlib_tensor(filename, dtype, shape):
    """
    读取zlib压缩的二进制文件，解压后转为numpy数组，再转为torch tensor。
    filename: 文件名
    dtype: np.int8 或 np.float32
    shape: 目标shape
    """
    with open(filename, 'rb') as f:
        data = f.read()
    raw = zlib.decompress(data)
    arr = np.frombuffer(raw, dtype=dtype)
    arr = arr.reshape((-1,) + shape)
    return torch.tensor(arr)


def copy_with_overwrite(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy2(src, dst)


def shuffle(tens_s):
    idx = torch.randperm(tens_s[0].shape[0])
    return (t[idx] for t in tens_s)
cs, ovs, ps, vs = [], [], [], []
if __name__ == "__main__":
    while os.path.exists(f'../training_data/{checkpoint}'):
        colors = load_zlib_tensor(f'../training_data/{checkpoint}/color.ttc', np.uint8, (9, 9))
        overalls = load_zlib_tensor(f'../training_data/{checkpoint}/over.ttc', np.float32, (15,))
        policies = load_zlib_tensor(f'../training_data/{checkpoint}/pols.ttc', np.float32, (4790,))
        values = load_zlib_tensor(f'../training_data/{checkpoint}/vals.ttc', np.int8, ())
        print(f'training checkpoint {checkpoint}')
        sz = colors.shape[0]
        assert overalls.shape[0] == sz and values.shape[0] == sz and policies.shape[0] == sz, f'size mismatch at checkpoint {checkpoint}'
        print(sz)
        cs.append(colors)
        ovs.append(overalls)
        ps.append(policies)
        vs.append(values)
        checkpoint += 1
        s += sz
        szs.append(s)
        f = open('checkpoint', 'w')
        f.write(str(s))
        f.close()
    colors, overalls, policies, values = shuffle((torch.cat(cs), torch.cat(ovs), torch.cat(ps), torch.cat(vs)))
        # valid_size = int(sz * VALID_RATIO)
        # valid_buffer.append(colors[:valid_size], overalls[:valid_size], policies[:valid_size], values[:valid_size])
        # train_buffer.append(colors[valid_size:], overalls[valid_size:], policies[valid_size:], values[valid_size:])
    train_buffer.append(colors, overalls, policies, values)
    train_buffer.save('./train_buffer/')

    if os.path.exists(f'../latest_state.pt'):
        copy_with_overwrite('../latest_state.pt', f'../oldData/latest{s}_state.pt')
        if os.path.exists(f'../latest_optim.pt'):
            copy_with_overwrite('../latest_optim.pt', f'../oldData/latest{s}_optim.pt')


    train(model, opt, train_buffer,device, '../')
    # while os.path.exists(f'../binary{checkpoint}.ttc'):
    #     def ren_move(name):
    #         global checkpoint
    #         s = szs[checkpoint]
    #         ns = szs[checkpoint + 1]
    #         os.rename(f'../{name}{checkpoint}.ttc', f'../{name}{s}-{ns}.ttc')
    #         if not os.path.exists(f'../oldData'):
    #             os.mkdir('../oldData')
    #         try: os.remove(f'../oldData/{name}{s}-{ns}.ttc')
    #         except: pass
    #         os.system(f'MOVE ../{name}{s}-{ns}.ttc ../oldData/{name}{s}-{ns}.ttc')
        # for i in range(len(bins)):
            # print(bins.shape)


        # ren_move('bin')
        # ren_move('over')
        # ren_move('pols')
        # ren_move('vals')
        # checkpoint += 1
