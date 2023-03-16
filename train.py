import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp

from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import sys
import clip
import ftfy
import regex as re
import html
import random
import tempfile
import tokenizers
import json
import random
import math

class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)
    
    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        return tokens


    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # tokens = self.captions_tokens[item]
        
        clip_tokens = self.pad_tokens(item)
        clip_tokens_77 = self.captions_tokens[item]
        return clip_tokens,clip_tokens_77

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.clip_tokenizer = clip.tokenize
        self.prefix_length = 10
        self.max_seq_len = 20
        with open(data_path, 'r') as f:
            self.captions = json.load(f)
        random.shuffle(self.captions)
        self.captions_tokens = []
        for caption in self.captions[:]:
            try:
                self.captions_tokens.append(torch.tensor(self.clip_tokenizer(caption)[0], dtype=torch.int64))
            except:
                continue
        print(len(self.captions_tokens))

    
class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class DeCap(nn.Module):

    def __init__(self,prefix_size: int = 512):
        super(DeCap, self).__init__()

        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        with open('./decoder_config.pkl','rb') as f:
            config = pickle.load(f)
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size,self.embedding_size))
        
    def forward(self, clip_features,gpt_tokens):
        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = embedding_clip.reshape(-1,1,self.embedding_size)
        embedding_cat = torch.cat([embedding_clip,embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out




def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


    

def train_decoder(dataset: ClipCocoDataset, args,
          lr: float = 1e-5, warmup_steps: int = 1000, output_dir: str = ".", output_prefix: str = ""):

    # device = torch.device('cuda:1')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.is_master = args.local_rank == 0

    # set the device
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda:'+str(args.local_rank))
    dist.init_process_group(backend='nccl', init_method='env://')
    SEED=42
    torch.cuda.manual_seed_all(SEED)
    
    model = DeCap()

    clip_model_type = "ViT-B/32"
    clip_model_name = clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    clip_model.eval()
    
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)
    model.to(device)
    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True
    )
    
    
    optimizer = AdamW(model.parameters(),lr=lr)
    
    sampler = DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=sampler,batch_size=batch_size,drop_last=True)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    
    
    for epoch in range(epochs):
        loss_token_save,ac_save= 0,0
        sys.stdout.flush()
        if args.is_master:
            print(f">>> Training epoch {epoch}")
            progress = tqdm(total=int(len(train_dataloader)/10), desc=output_prefix)

        dist.barrier()
        for idx,(clip_tokens,clip_tokens_77) in enumerate(train_dataloader):
            clip_tokens,clip_tokens_77 = clip_tokens.to(device),clip_tokens_77.to(device)
            
            with torch.no_grad():
                feature_text = clip_model.encode_text(clip_tokens_77)
                feature_text /= feature_text.norm(dim=-1, keepdim=True)
                # feature_text = noise_injection(feature_text,device = device)
            # shape = clip_tokens.shape
            # arr = torch.where(torch.rand(shape) < 0.9, torch.ones(shape), torch.zeros(shape)).to(device).long()

            outputs = model(feature_text.float(),clip_tokens)
            logits = outputs
            
            logits = logits.logits

            logits = logits[:,: -1]
            clip_tokens = clip_tokens.flatten()
            logits = logits.reshape(-1, logits.shape[-1])
            
            loss_token = loss_ce(logits, clip_tokens)
            ac=((logits.argmax(1)==clip_tokens)*(clip_tokens>0)).sum()/(clip_tokens>0).sum()
            optimizer.zero_grad()
            loss_all = loss_token
            loss_all.backward()
            optimizer.step()
            scheduler.step()
            if args.is_master:
                
                if(idx+1) %10 ==0:
                    progress.set_postfix({"loss_token": loss_token_save/10.0,"acc_token":ac_save/10.0})
                    progress.update()
                    loss_token_save,ac_save= 0,0
                else:
                    loss_token_save += loss_token.item()
                    ac_save += ac.item()

        if args.is_master:
            log_dir = './log/'+args.dataset+'.txt'
            with open(log_dir,'a+') as f:
                f.writelines('epoch ' +str(epoch) +': '+ progress.postfix+'\r\n')
            progress.close()
            if epoch % args.save_every == 0 or epoch == epochs - 1:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
                )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/doubleMTA_lr1e5_ignore0.pkl')
    parser.add_argument('--out_dir', default='./coco_model')
    parser.add_argument('--prefix', default='./coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--dataset', default='coco', help='coco or cc3m or bookcorpus')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=1)
    parser.add_argument('--prefix_length_clip', type=int, default=1)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.') 
    args = parser.parse_args()
    prefix_length = args.prefix_length
    dataset = ClipCocoDataset('data/'+args.dataset+'_train.json', prefix_length, normalize_prefix=args.normalize_prefix)

    train_decoder(dataset, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()

