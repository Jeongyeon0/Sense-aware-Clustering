import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from loader import wsdDataset, pad, tokenizer
import os
import numpy as np
import torch.nn.functional as F
import sys
import argparse
from transformers import AutoTokenizer
import transformers
tokenizer = AutoTokenizer.from_pretrained('roberta-large')

np.random.seed(1314)
torch.manual_seed(1314)
counter = 0 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_dict(restrict_file_name, recover_file_name, sense_tag_file_name, pos_tag_file_name):
    recover_dict={}
    restrict_tag={}
    sense_vocab=['<pad>','00']
    pos_vocab=['<pad>','00']

    for line in open('new_dict/'+restrict_file_name,'r').readlines():
        line = line.strip()
        line = line.split(' ')
        restrict_tag[line[0]]=line[1:]

    for line in open('new_dict/'+recover_file_name,'r').readlines():
        line = line.strip()
        line = line.split()
        recover_dict[line[0].split('%')[0], line[1]]=line[0]

    for line in open('new_dict/'+sense_tag_file_name,'r').readlines():
        line = line.strip()
        sense_vocab.append(line)
    sense2idx = {tag: idx for idx, tag in enumerate(sense_vocab)}
    idx2sense = {idx: tag for idx, tag in enumerate(sense_vocab)}
    
    for line in open('new_dict/'+pos_tag_file_name,'r').readlines():
        line = line.strip()
        pos_vocab.append(line)
    pos2idx = {tag: idx for idx, tag in enumerate(pos_vocab)}
    idx2pos = {idx: tag for idx, tag in enumerate(pos_vocab)}

    return recover_dict, restrict_tag, sense_vocab, sense2idx, idx2sense, pos_vocab, pos2idx, idx2pos



def train(model, iterator, optimizer, criterion, scheduler):
    model.train()
    for i, batch in enumerate(iterator):
        #words, x, is_heads, tags, y,  seqlens = batch
        words, sentence, is_heads, senses, sense_tags, POS, pos_tags, IDS, seqlens = batch
        _y = senses # for monitoring
        y = sense_tags.to(device)
        optimizer.zero_grad()
        logits = model(sentence) # logits: (N, T, VOCAB), y: (N, T)
        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i==0:
            print("=====sanity check======")
            print("words:", words[0])
            print("x:", sentence.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", tokenizer.convert_ids_to_tokens(sentence.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            #print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", sense_tags[0])
            print("seqlen:", seqlens[0])
            print("=======================")


        if i%500==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")


def eval(model, iterator, path, fname, epoch, maxf1, sense_tag_file, restrict_tag, recover_dict):
    model.eval()
    temp_file=sense_tag_file+'.temp'
    Words, Is_heads, Tags, Y, Y_hat, IDS  = [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            #words, x, is_heads, tags, y, seqlens = batch
            words, sentence, is_heads, senses, sense_tags, pos, pos_tags, id_tags, seqlens = batch
            all_labels_mask = []
            posit=0
            for word, pos_, ids in zip(words[0].split(' '), pos[0].split(' '), id_tags[0].split(' ')):
                #참고사항: "be" 단어의 주석 문제로, target 단어임을 나타내는 sSenses를 사용 ->평가문제임
                category = pos_
                #pos_info = pos[0].split(' ')[posit]
                if posit<2:
                    tokens = tokenizer.tokenize(word) if word not in ("<s>", "</s>") else [word]
                else:
                    tokens = tokenizer.tokenize(" "+word) if word not in ("<s>", "</s>") else [word]
                posit+=1
                piece_num = len(tokens)
                probability = []
                if word in restrict_tag.keys():
                    # 동음이의어 여부를 확인하기 위하여 lower() 기능을 이용하여 소문자 단어로 확인한다.
                    for sense in restrict_tag[word.lower()]:
                        if sense.split(':')[0]==category:
                            probability.append(sense2idx[sense])
                    #probability.append(sense2idx['00'])

                else:
                    probability.append(sense2idx['00'])

                for k in range(piece_num):
                    label_mask = [float('-inf')] * len(SENSE_VOCAB)
                    for idx in probability:
                        label_mask[idx] = 0
                    all_labels_mask.append(label_mask)
            
            all_labels_mask = torch.tensor(all_labels_mask, dtype=torch.float).to(device)
            
            logits = model(sentence)  # logits: (N, T, VOCAB), y: (N, T)
            y_hat = logits + all_labels_mask
            y_hat = y_hat.detach()
            y_hat = y_hat.argmax(-1)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(senses)
            Y.extend(sense_tags.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            IDS.extend(id_tags)


    ## gets results and save
    with open(temp_file, 'w') as fout:
        for ids, words, is_heads, tags, y_hat in zip(IDS, Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2sense[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for i, w, t, p in zip(ids.split()[1:-1], words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{i} {w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true =  np.array([sense2idx[line.split()[2]] for line in open(temp_file, 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([sense2idx[line.split()[3]] for line in open(temp_file, 'r').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    if fname=='dev':
        return precision, recall, f1

    if not os.path.isdir(path+'/'+str(epoch)):
        os.mkdir(path+'/'+str(epoch))
    final = path+'/'+str(epoch)+'/'+fname+".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    pred_file = open(path+'/'+str(epoch)+'/'+fname+'.pred','w')
    with open(final, 'w') as fout:
        result = open(temp_file, "r").readlines()
        for line in result:
            if line=='\n':
                fout.write('\n')
                continue
            line = line.strip()
            line = line.split(' ')
            fout.write(f"{line[0]} {line[1]} {line[2]} {line[3]}\n")
            if line[0]!='wf':
                pred_file.write(f"{line[0]} {recover_dict[line[1], line[3]]}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")

    os.remove(temp_file)

    print("precision=%.4f"%precision)
    print("recall=%.4f"%recall)
    print("f1=%.5f"%f1)
    return precision, recall, f1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--trainset", type=str, default="senseval3/train.conll")
    parser.add_argument("--validset", type=str, default="senseval3/test.conll")
    parser.add_argument("--restrict_file", type=str, default=None)
    parser.add_argument("--recover_file", type=str, default=None)
    parser.add_argument("--sense_tag_file", type=str, default=None)
    parser.add_argument("--pos_tag_file", type=str, default=None)
    #가중치 경로
    hp = parser.parse_args()


    
    recover_dict, restrict_tag, SENSE_VOCAB, sense2idx, idx2sense, POS_VOCAB, pos2idx, idx2pos = load_dict(hp.restrict_file, hp.recover_file, hp.sense_tag_file, hp.pos_tag_file)

    model = Net(hp.top_rnns, len(SENSE_VOCAB), 0.1, device, hp.finetuning).cuda()
    if hp.train:
        train_dataset = wsdDataset(hp.trainset, sense2idx, pos2idx)
        eval_dataset = wsdDataset(hp.validset+'dev.conll', sense2idx, pos2idx)
        senseval2_dataset = wsdDataset(hp.validset+'senseval2_corpus.conll', sense2idx, pos2idx)
        senseval3_dataset = wsdDataset(hp.validset + 'senseval3_corpus.conll', sense2idx, pos2idx)
        semeval2007_dataset = wsdDataset(hp.validset + 'semeval2007_corpus.conll', sense2idx, pos2idx)
        semeval2013_dataset = wsdDataset(hp.validset + 'semeval2013_corpus.conll', sense2idx, pos2idx)
        semeval2015_dataset = wsdDataset(hp.validset + 'semeval2015_corpus.conll', sense2idx, pos2idx)
        train_iter = data.DataLoader(dataset=train_dataset,
                                     batch_size=hp.batch_size,
                                     shuffle=True,
                                     num_workers=0,
                                     collate_fn=pad)

        eval_iter = data.DataLoader(dataset=eval_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=pad)

        senseval2_iter = data.DataLoader(dataset=senseval2_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=pad)

        senseval3_iter = data.DataLoader(dataset=senseval3_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=pad)

        semeval2007_iter = data.DataLoader(dataset=semeval2007_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0,
                                         collate_fn=pad)

        semeval2013_iter = data.DataLoader(dataset=semeval2013_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0,
                                         collate_fn=pad)

        semeval2015_iter = data.DataLoader(dataset=semeval2015_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=0,
                                           collate_fn=pad)
        adam_epsilon=1e-8
        #optimizer = optim.Adam(model.parameters(), lr = hp.lr)
        optimizer = optim.AdamW(model.parameters(), lr = hp.lr, eps=adam_epsilon)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=(len(train_dataset)/hp.batch_size)*hp.n_epochs)
        criterion = nn.CrossEntropyLoss()

        maxf1 = 0.0
        for epoch in range(1, hp.n_epochs+1):
            train(model, train_iter, optimizer, criterion, scheduler)
            avg_f1=0.0
            print(f"=========eval at epoch={epoch}=========")
            if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
            fname = str(epoch)
            precision, recall, f1 = eval(model, eval_iter, hp.logdir,'dev', epoch, maxf1, hp.sense_tag_file, restrict_tag, recover_dict)
            if f1 > maxf1:
                counter = 0
                maxf1 = f1
                torch.save(model, f"{hp.logdir}/best.pt")
                torch.save(model.state_dict(), f"{hp.logdir}/best_dict.pt")
                print(f"weights were saved to {hp.logdir}/best.pt")

                precision, recall, f1 = eval(model, senseval2_iter, hp.logdir, 'senseval2', epoch, maxf1, hp.sense_tag_file, restrict_tag, recover_dict)

                precision, recall, f1 = eval(model, senseval3_iter, hp.logdir, 'senseval3', epoch, maxf1, hp.sense_tag_file, restrict_tag, recover_dict)

                precision, recall, f1 = eval(model, semeval2007_iter, hp.logdir, 'semeval2007', epoch, maxf1, hp.sense_tag_file, restrict_tag, recover_dict)

                precision, recall, f1 = eval(model, semeval2013_iter, hp.logdir, 'semeval2013', epoch, maxf1, hp.sense_tag_file, restrict_tag, recover_dict)

                precision, recall, f1 = eval(model, semeval2015_iter, hp.logdir, 'semeval2015', epoch, maxf1, hp.sense_tag_file, restrict_tag, recover_dict)



