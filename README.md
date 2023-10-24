# CPSC 8470 Project - Phase 2

## Requirements

- Python 3.6
- PyTorch > 1.0
- Pandas
- Numpy
- Tqdm


## Run

Parameter Configuration:

- max_len: the max length of a sample, default for `170`
- early_stop: default for `5`
- epoch: training epoches, default for `50`
- aug_prob: probability of augmentation (mask and drop), default for `0.1`
- gpu: the index of gpu you will use, default for `0`
- lr: learning_rate, default for `0.0001`
- model_name: model_name within `bigru, bigru_endef, bert, bert_endef, bertemo, bertemo_endef, eann, eann_endef, mdfend, mdfend_endef`, default for `bigru_endef`

You can run this code through:

```powershell
# Reset several parameters
python main.py --gpu 1 --lr 0.0001 --model_name bigru
```


## Reference
- https://arxiv.org/pdf/2204.09484.pdf
