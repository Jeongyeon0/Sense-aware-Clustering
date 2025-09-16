import numpy as np
import torch
from torch.utils import data

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('roberta-large')


class wsdDataset(data.Dataset):
    def __init__(self, fpath, sense2idx, pos2idx):
        """
        fpath: [train|valid|test].txt
        """
        entries = open(fpath, 'r').read().strip().split("\n\n")
#        sents = [] # list of lists
        self.sense2idx=sense2idx
        self.pos2idx=pos2idx

        id_li, sents, senses_li, pos_li = [], [], [], [] # list of lists
        for entry in entries:
            ids = [line.split()[0] for line in entry.splitlines()]
            words = [line.split()[1] for line in entry.splitlines()]
            senses = ([line.split()[2] for line in entry.splitlines()])
            pos = ([line.split()[3] for line in entry.splitlines()])
            id_li.append(["<pad>"] + ids + ["<pad>"])
            sents.append(["<s>"] + words + ["</s>"])
            senses_li.append(["<pad>"] + senses + ["<pad>"])
            pos_li.append(["<pad>"] + pos + ["<pad>"])
        self.sents, self.senses_li, self.pos_li, self.id_li = sents, senses_li, pos_li, id_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, senses, POS, IDS = self.sents[idx], self.senses_li[idx], self.pos_li[idx], self.id_li[idx] # words, tags: string list
#        words = self.sents[idx] # words, tags: string list
        # We give credits only to the first piece.
        sentence, sense_tags, pos_tags, id_tags  = [], [], [], [] # list of ids, x is sentence, y is sense
        is_heads = [] # list. 1: the token is the first piece of a word
#        for w in words:
        cnt=0
        for word, sense, pos, id_tag in zip(words, senses, POS, IDS):
            if word in ("<s>", "</s>") or (cnt < 2): 
                tokens = tokenizer.tokenize(word)
                cnt+=1
            else:
                tokens = tokenizer.tokenize(' ' + word)
#            print(tokens)
            xx = tokenizer.convert_tokens_to_ids(tokens)
#           print(xx)
            is_head = [1] + [0]*(len(tokens) - 1)

            sense = [sense] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [self.sense2idx[each] for each in sense]  # (T,)

            pos = [pos] + ["<pad>"] * (len(tokens) - 1)
            pp = [self.pos2idx[each] for each in pos]
            ids = [id_tag] + ["<pad>"] * (len(tokens) - 1)

            sentence.extend(xx)
            is_heads.extend(is_head)
            sense_tags.extend(yy)
            pos_tags.extend(pp)
            id_tags.extend(ids)

        assert len(sentence)==len(sense_tags)==len(is_heads), f"len(x)={len(sentence)}, len(y)={len(sense_tags)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(sense_tags)

        # to string
        words = " ".join(words)
        senses = " ".join(senses)
        POS = " ".join(POS)
        IDS = " ".join(IDS)
        return words, sentence, is_heads, senses, sense_tags, POS, pos_tags, IDS, id_tags, seqlen


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    senses = f(3)
    POS = f(5)
    IDS = f(7)
    
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [1] * (seqlen - len(sample[x])) for sample in batch] # 1: <pad> in roberta-base
    x = f(1, maxlen)
    y = f(4, maxlen)
    z = f(6, maxlen)
    u = f(8, maxlen)
    f = torch.LongTensor

#    return words, f(x), is_heads, seqlens
    return words, f(x), is_heads, senses, f(y), POS, f(z), IDS, seqlens


