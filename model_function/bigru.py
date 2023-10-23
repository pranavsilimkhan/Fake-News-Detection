import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader

class BiGRUModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, num_layers):
        super(BiGRUModel, self).__init__()
        self.fea_size = emb_dim
        self.bert = BertModel.from_pretrained('bert-base-uncased').requires_grad_(False)
        self.embedding = self.bert.embeddings
        
        self.rnn = nn.GRU(input_size = emb_dim,
                          hidden_size = self.fea_size,
                          num_layers = num_layers, 
                          batch_first = True, 
                          bidirectional = True)

        input_shape = self.fea_size * 2
        self.attention = MaskAttention(input_shape)
        self.mlp = MLP(input_shape, mlp_dims, dropout)
    
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        feature = self.embedding(inputs)
        feature, _ = self.rnn(feature)
        feature, _ = self.attention(feature, masks)
        output = self.mlp(feature)
        return torch.sigmoid(output.squeeze(1))


def test(dataloader, model, config):
    pred = []
    label = []
    model.eval()
    data_iter = tqdm.tqdm(dataloader)
    for step_n, batch in enumerate(data_iter):
        with torch.no_grad():
            batch_data = data2gpu(batch, config['use_cuda'])
            batch_label = batch_data['label']
            batch_pred = model(**batch_data)

            label.extend(batch_label.detach().cpu().numpy().tolist())
            pred.extend(batch_pred.detach().cpu().numpy().tolist())
    
    return metrics(label, pred)

def train_bert(config, logger):
    save_path = os.path.join(config['save_param_dir'], config['model_name'])
    if os.path.exists(save_path):
        save_param_dir = save_path
    else:
        save_param_dir = os.makedirs(save_path)


    if(logger):
        logger.info('start training......')
    model = BiGRUModel(config['emb_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'])
    if config['use_cuda']:
        model = model.cuda()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    recorder = Recorder(config['early_stop'])
    val_loader = get_dataloader(config['root_path'] + 'val.json', config['max_len'], config['batchsize'], shuffle=False, use_endef=False aug_prob=config['aug_prob'])

    for epoch in range(config['epoch']):
        model.train()
        train_loader = get_dataloader(config['root_path'] + 'train.json', config['max_len'], config['batchsize'], shuffle=True, use_endef=False, aug_prob=config['aug_prob'])
        train_data_iter = tqdm.tqdm(train_loader)
        avg_loss = Averager()

        for step_n, batch in enumerate(train_data_iter):
            batch_data = data2gpu(batch, config['use_cuda'])
            label = batch_data['label']

            pred = model(**batch_data)
            loss = loss_fn(pred, label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.add(loss.item())
        print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

        results = test(val_loader, model, config)
        mark = recorder.add(results)
        if mark == 'save':
            torch.save(model.state_dict(),
                os.path.join(save_path, 'parameter_bert.pkl'))
        elif mark == 'esc':
            break
        else:
            continue
    model.load_state_dict(torch.load(os.path.join(save_path, 'parameter_bigru.pkl')))

    test_future_loader = get_dataloader(config['root_path'] + 'test.json', config['max_len'], config['batchsize'], shuffle=False, use_endef=False, aug_prob=config['aug_prob'])
    future_results = test(test_future_loader, model, config)
    if(logger):
        logger.info("start testing......")
        logger.info("test score: {}.".format(future_results))
        logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(config['lr'], config['aug_prob'], future_results['metric']))
    print('test results:', future_results)
    return future_results, os.path.join(save_path, 'parameter_bigru.pkl')