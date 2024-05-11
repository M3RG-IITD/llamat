import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

mem_in_mb = int(sys.argv[1])
gpus_to_populate = sys.argv[2].split(',')
gpus_to_populate = [int(i) for i in gpus_to_populate]

input_size = 784
memory_limit = mem_in_mb * 1024 * 1024

bytes_per_param = 4
approx_params = memory_limit // bytes_per_param

def find_max_hidden_size(input_size, approx_params):
    for hidden_size in range(1, approx_params):
        total_params = (input_size + 1) * hidden_size + (hidden_size + 1) * input_size
        if total_params > approx_params:
            return hidden_size - 1
    return 1

hidden_size = find_max_hidden_size(input_size, approx_params)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for gpu in gpus_to_populate:
            self.layers.append(nn.Linear(input_size, hidden_size).cuda(gpu))
            self.layers.append(nn.Linear(hidden_size, input_size).cuda(gpu))
    
    def forward(self, x):
        for _, gpu in enumerate(gpus_to_populate):
            x = x.cuda(gpu)
            x = torch.relu(self.layers[_ * 2](x))
            x = self.layers[_*2+1](x)
        return x

model = MLP(input_size, hidden_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
inputs = torch.randn(64, input_size)
targets = torch.randint(0, input_size, (64,))

inputs = torch.randn(64, input_size)
targets = torch.randint(0, input_size, (64,))

print("Utilising the gpus :)")
while True:
    outputs = model(inputs)

