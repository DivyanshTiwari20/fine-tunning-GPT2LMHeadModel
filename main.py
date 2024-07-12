from transformers import GPT2LMHeadModel, GPT2Tokenizer 
from chatData import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch

def train(chatData,model, optim):

    epochs=10

    for i in tqdm(range(epochs)):
        for X, a in chatData:
            optim.zero_grad()
            loss = model(X, attention_mask=a, lables=X).loss
            loss.backward()
            optim.step()
        torch.save(model.state_dict(), "model_state.pt")

def infer(inp):
    inp = "<startofstr>"+inp+"<bots>:"
    inp = tokenizer(inp)
    output = model.generate(**inp)
    output = tokenizer.decode(output[0])
    return output

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<pad>",
                              "bos_token":"<startofstring>",
                              "eos_token": "<endofstring>"})

tokenizer.add_tokens("<bots>:")

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_position_embeddings(len(tokenizer))

# print(Tokenizer.decode(model.generate(**Tokenizer("hey i was good at basketball but", return_tensors="pt" ))[0]))
# print(Tokenizer.decode(model.generate(**Tokenizer("hey my name is divyansh and ", return_tensors="pt" ))[0]))


ChatData = ChatData("chat_data.json",tokenizer)

model.train()

optim= Adam(model.parameters())