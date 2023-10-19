import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='bigru_endef')
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--aug_prob', type=float, default=0.1)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--root_path', default='/data/zhuyc/fake-news-detection/data/temporal_dedup_gossipcop/')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--gpu', default='0')
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--save_log_dir', default= './logs')
parser.add_argument('--save_param_dir', default= './param_model')
parser.add_argument('--param_log_dir', default = './logs/param')



import logging
import os
import json

from models.bigru import Trainer as BiGRUTrainer
from models.bert import Trainer as BertTrainer
from models.eann import Trainer as EANNTrainer
from models.mdfend import Trainer as MDFENDTrainer
from models.bertemo import Trainer as BertEmoTrainer
from models.bigruendef import Trainer as BiGRU_ENDEFTrainer
from models.bertendef import Trainer as BERT_ENDEFTrainer
from models.bertemoendef import Trainer as BERTEmo_ENDEFTrainer
from models.eannendef import Trainer as EANN_ENDEFTrainer
from models.mdfendendef import Trainer as MDFEND_ENDEFTrainer
import nltk


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from grid_search import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


from model_function.bert import *

print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {}'.format(args.lr, args.model_name, args.batchsize, args.epoch, args.gpu))


config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'root_path': args.root_path,
        'aug_prob': args.aug_prob,
        'weight_decay': 5e-5,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': 0.2}
            },
        'emb_dim': args.emb_dim,
        'lr': args.lr,
        'epoch': 15,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_log_dir': args.save_log_dir,
        'save_param_dir': args.save_param_dir,
        'param_log_dir': args.param_log_dir
}



# grid_search.py

def frange(x, y, jump):
  while x < y:
      x = round(x, 8)
      yield x
      x += jump


def getFileLogger(log_file):
    logger = logging.getLogger()
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def config2dict():
    config_dict = {}
    for k, v in configinfo.items():
        config_dict[k] = v
    return config_dict





param_log_dir = config['param_log_dir']
if not os.path.exists(param_log_dir):
    os.makedirs(param_log_dir)
param_log_file = os.path.join(param_log_dir, config['model_name'] +'_'+ 'param.txt')
logger = getFileLogger(param_log_file)  

train_param = {
    'lr': [config['lr']] * 10,
}
print(train_param)
param = train_param
best_param = []
json_path = './logs/json/' + config['model_name'] + str(config['aug_prob']) + '.json'
json_result = []
for p, vs in param.items():
    best_metric = {}
    best_metric['metric'] = 0
    best_v = vs[0]
    best_model_path = None
    vs = [0.0001]
    # for i, v in enumerate(vs):
    #     config['lr'] = v
    #     if config['model_name'] == 'eann':
    #         trainer = EANNTrainer(config)
    #     elif config['model_name'] == 'bertemo':
    #         trainer = BertEmoTrainer(config)
    #     elif config['model_name'] == 'bigru':
    #         trainer = BiGRUTrainer(config)
    #     elif config['model_name'] == 'mdfend':
    #         trainer = MDFENDTrainer(config)
    #     elif config['model_name'] == 'bert':
    #         trainer = BertTrainer(config)
    #     elif config['model_name'] == 'bigru_endef':
    #         trainer = BiGRU_ENDEFTrainer(config)
    #     elif config['model_name'] == 'bert_endef':
    #         trainer = BERT_ENDEFTrainer(config)
    #     elif config['model_name'] == 'bertemo_endef':
    #         trainer = BERTEmo_ENDEFTrainer(config)
    #     elif config['model_name'] == 'eann_endef':
    #         trainer = EANN_ENDEFTrainer(config)
    #     elif config['model_name'] == 'mdfend_endef':
    #         trainer = MDFEND_ENDEFTrainer(config)

        # nltk.download('punkt')
        # metrics, model_path = trainer.train(logger)
    metrics, model_path = train_bert(config, logger)
    json_result.append(metrics)
    if metrics['metric'] > best_metric['metric']:
        best_metric['metric'] = metrics['metric']
        best_v = v
        best_model_path = model_path
    best_param.append({p: best_v})
    print("best model path:", best_model_path)
    print("best metric:", best_metric)
    logger.info("best model path:" + best_model_path)
    logger.info("best param " + p + ": " + str(best_v))
    logger.info("best metric:" + str(best_metric))
    logger.info('--------------------------------------\n')
with open(json_path, 'w') as file:
    json.dump(json_result, file, indent=4, ensure_ascii=False)
