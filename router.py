from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch
import os
from tqdm import tqdm
import json

class train_config():
    def __init__(self):
        self.lr_encoder = 2e-5
        self.lr_classifier = 1e-4
        self.epoch = 3
        self.outdir = ""

MAP_category2id = {
    "left_right":0,
    "mcq":1,
    "distance":2,
    "count":3,
}

MAP_id2category = {MAP_category2id[x]:x for x in MAP_category2id.keys()}


class dataset(Dataset):
    def __init__(self, encoder, tokenizer, json_dir):
        with open(json_dir, 'r') as f:
            data = json.load(f)
            self.train_data = [[x["conversations"][0]["value"], MAP_category2id[x["category"]]] for x in data]
        self.encoder = encoder
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        [text, category_id] = self.train_data[idx]

        return text, category_id




class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ROUTER(nn.Module):
    def __init__(self):
        super().__init__()
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.encoder = RobertaModel.from_pretrained("roberta-base").to(self.device)
        self.classifier = Classifier(768).to(self.device)

    def forward(self, text, train_encoder = False):

        if not train_encoder:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors = "pt").to(self.device)
                
                outputs = self.encoder(**inputs)
                cls_embedding = outputs.last_hidden_state[:,0,:]

        else:
            inputs = self.tokenizer(text, return_tensors = "pt")
            outputs = self.encoder(**inputs)
            cls_embedding = outputs.last_hidden_state[:,0,:]
        
        return self.classifier(cls_embedding)
    
    def classify(self, text):
        
        with torch.no_grad():
            logit = self(text)
            output = torch.argmax(logit, 1, False).tolist()[0]
        return output
    
    def save_checkpoint(self, dir, ver = ""):

        try:
            os.makedirs(dir, exist_ok= True)
        except:
            pass

        torch.save(self.state_dict(), os.path.join(dir, f"router_{ver}.pth"))

    def train(self, train_dataloader, val_dataloader, train_config, train_encoder = False):

        criterion = nn.CrossEntropyLoss();
        optimizer = torch.optim.AdamW([
            {"params": self.encoder.parameters(), "lr": train_config.lr_encoder},
            {"params": self.classifier.parameters(), "lr": train_config.lr_classifier}
        ], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, train_config.epoch * len(train_dataloader), 1e-6
        )

        for epoch in range(train_config.epoch):
            total_loss = 0
            for batch in tqdm(train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                output = self.encoder(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                )
                cls = output.last_hidden_state[:, 0, :]
                logits = self.classifier(cls)
                loss = criterion(logits, labels)

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            print(f"Epoch: {epoch}, loss = {total_loss / len(train_dataloader)}")
            self.save_checkpoint(train_config.outdir, ver = f"ep{epoch}", save_encoder = True)

            # validation
            val_correct = 0
            val_total = 0
            for batch in tqdm(val_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                with torch.no_grad():
                    output = self.encoder(
                        input_ids = input_ids,
                        attention_mask = attention_mask
                    )
                    cls = output.last_hidden_state[:, 0, :]
                    logits = self.classifier(cls)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
            val_acc = val_correct / val_total
            
            print(f"Validation Accuracy: {val_acc:.4f}")

    def test(self, dataloader):

        val_correct = 0
        val_total = 0
        import time
        start_time = time.time()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            with torch.no_grad():
                output = self.encoder(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                )
                cls = output.last_hidden_state[:, 0, :]
                logits = self.classifier(cls)
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
        val_acc = val_correct / val_total
        end_time = time.time()
        print(f"val_correct: {val_correct}")
        print(f"val_total: {val_total}")
        print(f"Inference {val_total} questions in {(end_time - start_time)//60} min {(end_time - start_time)%60} sec")
        print(f"Average: {(end_time - start_time)/val_total} sec each")


                
        
if __name__ == "__main__":
    text = "From this viewpoint, does the pallet <mask> appear on the right-hand side of the pallet <mask>?"
    gt = "left_right"

    router = ROUTER()
    router.load_state_dict(torch.load("checkpoint/router/router_ep2.pth", weights_only= True))

    id = router.classify(text)

    print(f"Q: {text}")
    print(f"pred: {MAP_id2category[id]}")
    print(f"gt: {gt}")

    import sys
    if len(sys.argv) > 1:
        print("Inference json file")

        json_dir = sys.argv[1]
        with open(json_dir, 'r') as f:
            data = json.load(f)
            train_data = [[x["conversations"][0]["value"], MAP_category2id[x["category"]]] for x in data]
        
        import time
        start_time = time.time()
        total = 0;
        acc = 0;
        for text, cate in tqdm(train_data):
            id = router.classify(text)
            pred = MAP_id2category[id]
            total+=1;
            if cate==id:
                acc+=1;
        print(f"accuracy: {acc / total}")
        end_time = time.time()
        if(acc==total):
            print("all correct")
        print(f"Inference {total} questions in {(end_time - start_time)//60} min {(end_time - start_time)%60} sec")
        print(f"Average: {(end_time - start_time)/total} sec each")


