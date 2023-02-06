


import torch
import torch.nn as nn
from torch.nn import functional as F



from gpt import *


device = torch.device('cuda')

model = GPTLanguageModel()
m = model.to(device)

# Load the model
m.load_state_dict(torch.load('model.pt'))


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))