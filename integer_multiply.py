import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from model import Transformer, ModelArgs
from datetime import datetime

# model init
model_args = dict(
    dim=256,
    n_layers=8,
    n_heads=8,
    n_kv_heads=8,
    vocab_size=2,
    multiple_of=32,
    max_seq_len=256,
    dropout=0,
)  # start with model_args from command line

# wandb logging
wandb_log = True
wandb_project = "integer_multiply_transformer"

device_type = 'cuda'
device = torch.device(device_type)
ptdtype = torch.bfloat16
batch_size = 2048
num_workers = 4
log_interval = 50
eval_interval = 1000
eval_samples = batch_size * 5
max_iters = 100000
train_samples = max_iters * batch_size
label_mask_input = True
operation = 'multiply'

init_from = None

warmup_iters = 200  # how many steps to warm up for
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
learning_rate = 1e-4
beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1
min_lr = 0
iters = 0

wandb_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

class Dataset(Dataset):
    def __init__(self, size, num_digits, result_num_digits, operation):
        self.size = size
        self.num_digits = num_digits
        self.operation = operation
        self.max_val = 2 ** num_digits - 1  # 最大值基于位数
        self.result_num_digits = result_num_digits

    def __len__(self):
        return self.size

    def _generate_valid_sample(self):
        a, b = np.random.randint(0, self.max_val + 1, 2)

        # 执行运算
        if self.operation == 'add':
            result = a + b
        elif self.operation == 'multiply':
            result = a * b
        else:
            raise ValueError("Unsupported operation. Choose 'add' or 'multiply'.")

        return a, b, result

    def _int_to_digits(self, x, bits):
        return list(reversed([int(d) for d in format(x, f'0{bits}b')]))

    def __getitem__(self, idx):
        a, b, result = self._generate_valid_sample()
        # print(a, b, result)

        # 将数值转换为数字序列
        a_digits = self._int_to_digits(a, self.num_digits)
        b_digits = self._int_to_digits(b, self.num_digits)
        result_digits = self._int_to_digits(result, self.result_num_digits)

        # 将输入和输出合并为一个序列，用于Transformer的处理
        input_digits = a_digits + b_digits
        return torch.tensor(input_digits, dtype=torch.int64), torch.tensor(result_digits, dtype=torch.int64)
    

gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)

if init_from is not None:
    print(f"load from {init_from}")
    ckpt = torch.load(init_from)
    model.load_state_dict(ckpt)

model.to(device)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

raw_model = model
model = torch.compile(model)

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

def eval_model(model, num_digits, result_num_digits):
    model.eval()
    test_dataset = Dataset(eval_samples, num_digits, result_num_digits, operation)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    acc_list = []
    all_acc_list = []
    print("eval model")
    for inputs, labels in tqdm(test_dataloader, total=len(test_dataloader)):
        with ctx:
            inputs = inputs.to(device)
            labels = labels.to(device)
            out = model.generate(inputs, result_num_digits, temperature=0)
            out = out[:, -result_num_digits:]
            acc = (out == labels).float().mean()
            all_acc = torch.all(out == labels, dim=1).float().mean()
            acc_list.append(acc.item())
            all_acc_list.append(all_acc.item())
            # if all_acc.item() < 1:
            #     print(torch.concat([inputs, labels, out], dim=1))

    model.train()
    return np.mean(acc_list), np.mean(all_acc_list)

def train_model(model, num_digits, result_num_digits):
    train_dataset = Dataset(train_samples, num_digits, result_num_digits, operation)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    global iters
    for i, (inputs, labels) in tqdm(enumerate(train_dataloader), initial=iters, total=len(train_dataloader)):
        lr = get_lr(iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        inputs = inputs.to(device)
        labels = labels.to(device)
        x = torch.concat([inputs, labels], dim=1)
        X = x[:, :-1].contiguous()
        if label_mask_input:
            Y = x.clone()
            Y[:, :num_digits*2] = -1
            Y = Y[:, 1:].contiguous()
        else:
            Y = x[:, 1:].contiguous()
        optimizer.zero_grad()
        with ctx:
            logits, loss = model(X, Y)
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        iters += 1
        if (i + 1) % log_interval == 0:
            print(f"Step {i} LR: {lr:.6f} Loss: {loss.item():.4f}")
            if wandb_log:
                wandb.log({"loss": loss.item(), "lr": lr}, step=iters)
        if (i + 1) % eval_interval == 0:
            acc, all_acc = eval_model(raw_model, num_digits, result_num_digits)
            print(f"Step {i} LR: {lr:.6f} Loss: {loss.item():.4f} Acc: {acc:.4f} All_Acc: {all_acc:.4f}")
            if wandb_log:
                wandb.log({"eval/acc": acc, "eval/all_acc": all_acc}, step=iters)

            if all_acc >= 1.0:
                print(f"success in {i} steps")
                break
    torch.save(model.state_dict(), 'model.pt')

# num_digits = 1
for num_digits in range(31, 32, 2):
    result_num_digits = num_digits + 2 if operation == 'add' else num_digits * 2 + 1
    print(f"num_digits {num_digits} result_num_digits {result_num_digits}")
    train_model(model, num_digits, result_num_digits)