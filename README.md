# KG-RuleGuider

Implementation of our EMNLP 2020 paper [Learning Collaborative Agents with Rule Guidance for Knowledge Graph Reasoning](https://arxiv.org/abs/2005.00571).

## Installation

Install PyTorch (>= 1.4.0) following the instructions on the [PyTorch](https://pytorch.org/).
Our code is written in Python3.

Run the following commands to install the required packages:
```
pip3 install -r requirements.txt
```

## Data Preparation

Unpack the data files:

```
unzip data.zip
```

It will generate three dataset folders in the ./data directory. In our experiments, the datasets used are: `fb15k-237`, `wn18rr` and `nell-995`.

## Rule Mining


## Training

1. Train embedding-based models:

```
./experiment-emb.sh configs/<dataset>-<model>.sh --train <gpu-ID>
```

2. Pretrain relation agent using top rules:

```
./experiment-pretrain.sh configs/<dataset>-rs.sh --train <gpu-ID> <rule-path> --model point.rs.<embedding-model>
```

3. Jointly train relation agent and entity agent with reward shaping

```
./experiment-rs.sh configs/<dataset>-rs.sh --train <gpu-ID> <rule-path> --model point.rs.<embedding-model> --checkpoint_path <pretrain-checkpoint-path>
```

Note:
* you can choose embedding models among `conve`, `complex`, and `distmult`.
* you have to pre-train the embedding-based models before pretraining relation agent or jointly training two agents.
* you can skip pretraining relation agent.
* make sure you set the file path pointers to the pre-trained embedding-based models correctly (example configuration file),
* use `--board <board-path>` to logs the training details, `--model <model-path>` to assign the directory in which checkpoints are saved, and `--checkpoint_path <checkpoint-path>` to load checkpoints.
* in joint training, you can use `--rule_ratio <ratio>` to specify the ratio between rule reward and hit reward.



## Evaluation

1. Evaluate embedding-based models:


```
./experiment-emb.sh configs/<dataset>-<model>.sh --inference <gpu-ID>
```

2. Evaluate the pretraining of relation agent :

```
./experiment-pretrain.sh configs/<dataset>-rs.sh --inference <gpu-ID> <rule-path> --model point.rs.<embedding-model> --checkpoint_path <pretrain-checkpoint-path>
```

3. Evaluate the final result:

```
./experiment-rs.sh configs/<dataset>-rs.sh --inference <gpu-ID> <rule-path> --model point.rs.<embedding-model> --checkpoint_path <checkpoint-path>
```

## Citation

If you find the repository helpful, please cite
```
@article{lei2020learning,
  title={Learning Collaborative Agents with Rule Guidance for Knowledge Graph Reasoning},
  author={Lei, Deren and Jiang, Gangrong and Gu, Xiaotao and Sun, Kexuan and Mao, Yuning and Ren, Xiang},
  journal={arXiv preprint arXiv:2005.00571},
  year={2020}
}
```