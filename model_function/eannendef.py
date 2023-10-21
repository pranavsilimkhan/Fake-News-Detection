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

class EANN_ENDEFModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout):
        super(EANN_ENDEFModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased').requires_grad_(False)
        self.embedding = self.bert.embeddings
        domain_num = 3

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.convs = cnn_extractor(feature_kernel, emb_dim)
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.classifier = MLP(mlp_input_shape, mlp_dims, dropout)
        self.domain_classifier = nn.Sequential(MLP(mlp_input_shape, mlp_dims, dropout, False), torch.nn.ReLU(),
                        torch.nn.Linear(mlp_dims[-1], domain_num))
        
        self.entity_convs = cnn_extractor(feature_kernel, emb_dim)
        self.entity_mlp = MLP(mlp_input_shape, mlp_dims, dropout)
        self.entity_net = torch.nn.Sequential(self.entity_convs, self.entity_mlp)
    
    def forward(self, alpha, **kwargs):
        inputs = kwargs['content']
        bert_feature = self.embedding(inputs)
        feature = self.convs(bert_feature)
        bias_pred = self.classifier(feature).squeeze(1)
        reverse = ReverseLayerF.apply
        domain_pred = self.domain_classifier(reverse(feature, alpha))

        entity = kwargs['entity']
        entity_feature = self.embedding(entity)
        entity_prob = self.entity_net(entity_feature).squeeze(1)

        return torch.sigmoid(0.9 * bias_pred + 0.1 * entity_prob), torch.sigmoid(entity_prob), domain_pred, torch.sigmoid(bias_pred)



def test(dataloader, model, config):
    pred = []
    label = []
    model.eval()
    data_iter = tqdm.tqdm(dataloader)
    for step_n, batch in enumerate(data_iter):
        with torch.no_grad():
            batch_data = data2gpu(batch, config['use_cuda'])
            batch_label = batch_data['label']
            _, __, ___, batch_pred = model(**batch_data, alpha =-1)

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
    model = EANN_ENDEFModel(config['emb_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'])
    if config['use_cuda']:
        model = model.cuda()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    recorder = Recorder(config['early_stop'])
    val_loader = get_dataloader(config['root_path'] + 'val.json', config['max_len'], config['batchsize'], shuffle=False, use_endef=True aug_prob=config['aug_prob'])

    for epoch in range(config['epoch']):
        model.train()
        train_loader = get_dataloader(config['root_path'] + 'train.json', config['max_len'], config['batchsize'], shuffle=True, use_endef=True, aug_prob=config['aug_prob'])
        train_data_iter = tqdm.tqdm(train_loader)
        avg_loss = Averager()
		alpha = max(2. / (1. + np.exp(-10 * epoch / self.config['epoch'])) - 1, 1e-1)

        for step_n, batch in enumerate(train_data_iter):
            batch_data = data2gpu(batch, config['use_cuda'])
            label = batch_data['label']
			domain_label = batch_data['year']

            pred,domain_pred = model(**batch_data, alpha = alpha)
            loss = loss_fn(pred, label.float())+ 0.2 * loss_fn(entity_pred, label.float())
			loss_adv = F.nll_loss(F.log_softmax(domain_pred, dim=1), domain_label)
            loss = loss + loss_adv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.add(loss.item())
        print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

        results = test(val_loader, model, config)
        mark = recorder.add(results)
        if mark == 'save':
            torch.save(model.state_dict(),
                os.path.join(save_path, 'parameter_eannendef.pkl'))
        elif mark == 'esc':
            break
        else:
            continue
    model.load_state_dict(torch.load(os.path.join(save_path, 'parameter_eannendef.pkl')))

    test_future_loader = get_dataloader(config['root_path'] + 'test.json', config['max_len'], config['batchsize'], shuffle=False, use_endef=True, aug_prob=config['aug_prob'])
    future_results = test(test_future_loader, model, config)
    if(logger):
        logger.info("start testing......")
        logger.info("test score: {}.".format(future_results))
        logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(config['lr'], config['aug_prob'], future_results['metric']))
    print('test results:', future_results)
    return future_results, os.path.join(save_path, 'parameter_eannendef.pkl')