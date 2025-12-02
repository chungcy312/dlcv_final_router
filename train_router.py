from router import ROUTER, MAP_category2id, MAP_id2category, train_config, dataset
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch
import argparse
import json
import os

parser = argparse.ArgumentParser()

batch_size = 256


parser.add_argument("--checkpoint_dir", required = False, default = "checkpoint/router")
parser.add_argument("--train_json", required = False, default = "DLCV_Final1/train.json")
parser.add_argument("--val_json", default = "DLCV_Final1/val.json", required = False)
parser.add_argument("--epoch", default = 3, required = False, type = int)

args = parser.parse_args()

try:
    os.makedirs(args.checkpoint_dir, exist_ok = True)
except:
    pass

def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    encoded = router.tokenizer(
        texts,
        padding="longest",
        truncation=True,
        return_tensors="pt"
    )

    encoded["labels"] = labels
    return encoded

router = ROUTER()

router.load_state_dict("checkpoint/router/encoder_ep0.pth", "checkpoint/router/classifier_ep0.pth")

# train_dataset = dataset(router.encoder, router.tokenizer, args.train_json)
val_dataset = dataset(router.encoder, router.tokenizer, args.val_json)

# train_dataloader = DataLoader(train_dataset, batch_size, True, collate_fn = collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size, False, collate_fn = collate_fn)

train_cfg = train_config()
train_cfg.outdir = args.checkpoint_dir
train_cfg.epoch = args.epoch

# router.train(train_dataloader, val_dataloader, train_cfg, True)
router.test(val_dataloader)