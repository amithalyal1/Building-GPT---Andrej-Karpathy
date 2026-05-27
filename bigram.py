import torch 
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
batch_size = 32 #Basically how many sequences will be processed in parallel (Earlier was it 4)
block_size = 8 #context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# -------------

torch.manual_seed(1337)

with open('input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
#We need it to be a list because we will be using the index of the characters to encode them, set does not have order and hence we cannot use the index of the characters in a set
vocab_size = len(chars)

#Create mapping to act as the tokenizer
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)} 
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

#Train test split
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    data = train_data if split =='train' else test_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    #We take i+1 in targets because we want to predict the next character given the previous char,
    # hence naturally there is an offset of 1 between the input and the target
    x, y =x.to(device), y.to(device)
    return x, y

@torch.no_grad() #Here the decorator is used to tell PyTorch that we do not need to compute gradients for the operations in this function, which can save memory and computations during evaluation
def estimate_loss():
    out = {}
    model.eval() 
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx) 
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)

        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):

    if iter %eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = 100)[0].tolist()))

#How can I get this .py to run? from terminal, I can't see any error but it doesn't run either, I am not sure if I am missing something. I have added the code to run the training loop and generate text at the end of the class definition, but it is not executing when I run the .py file. Do I need to add a main function or something?
# 
