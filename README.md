## Causally-Aware Representations of Multi-Agent Interactions

<p align="center">
  <img src="docs/background.png" width="600">
</p>

> TL;DR: we investigate causal representation learning in the multi-agent context, from computational formalisms to controlled simulations to real-world applications.
> 1. we cast doubt on the notion of non-causal robustness in the previous benchmark, revealing that recent representations tend to underestimate indirect causal effects
> 2. we introduce a causal regularization approach, including a contrastive and a ranking variant, which leads to higher causal awareness and out-of-distribution robustness
> 3. we propose a sim-to-real causal transfer framework, which enables causally-aware representation learning in practical settings even without real-world annotations

<p align="center">
  <img src="docs/method.png" width="800">
</p>

## Getting Started

To install requirements:
```
pip install -r requirements.txt
```

Our diagnostic dataset can be downloaded from [Google drive](https://drive.google.com/file/d/1j4heKWyUia4hYhKY6pjLteoN9o0kfeKe/view?usp=drive_link). It comprises 20k training scenes, 2k in-distribution test scenes, and 2k out-of-distribution test scenes, with the following directory structure:
```

─── dataset-name
   ├── train
   │   ├── scene_0.pkl
   │   ├── scene_1.pkl
   │   ├── ...
   │   └── scene_19999.pkl
   └── val
       ├── scene_0.pkl
       ├── scene_1.pkl
       ├── ...
       └── scene_1999.pkl
```

## Synthetic Experiments

### Baselines

To train the [AutoBots](https://openreview.net/forum?id=Dup_dDqkZC5) baseline on our diagnostic dataset:
```
python train.py --exp-id baseline --save-dir <results directory, e.g., ./ > --dataset-path <path to synth dataset> --evaluate_causal
```

To run the [data augmentation](https://arxiv.org/abs/2207.03586) baseline:
```
python train.py --exp-id baseline --save-dir <results directory, e.g., ./> --dataset-path <path to synth dataset> --evaluate_causal --reg-type augment
```

### Causal Regularization

For a fair and efficient comparision between different methods, we fine-tune the same pre-trained model in our experiments.

To run the contrastive regularization:
```
python train.py --exp-id baseline --save-dir <results directory, e.g., ./> --dataset-path <path to synth dataset> --evaluate_causal --reg-type contrastive \
        --weight-path <path to the last ckpt of baseline model, e.g., ./results/Autobot_ego_regType:None_baseline_s1/models_700.pth> --start-epoch 700
```

To run the ranking regularization:
```
python train.py --exp-id baseline --save-dir <results directory, e.g., ./> --dataset-path <path to synth dataset> --evaluate_causal --reg-type ranking \
        --weight-path <path to the last ckpt of baseline model, e.g., ./results/Autobot_ego_regType:None_baseline_s1/models_700.pth> --start-epoch 700
```

### Evaluation

To evaluate on OOD sets:
```
python evaluate.py --models-path <path to the model> --dataset-path <path to the ood dataset>
```

## Real-world Experiments

### Baselines

To train the [AutoBots](https://openreview.net/forum?id=Dup_dDqkZC5) baseline on the ETH-UCY dataset:
```
python train.py --exp-id <output tag> --dataset s2r --reg-type contrastive --dataset-path <path to the ETH-UCY dataset> --num-encoder-layers 1 --num-decoder-layers 1 --num-epochs 50 --learning-rate-sched 10 20 30 40 50 --low-data 1.0 --dataset-path-real <path to the ETH-UCY dataset> --dataset-path-synth <path to the synthetic dataset> --contrastive-weight 0.0 --save-dir <directory for saving results> 
```

To run the vanilla sim2real, i.e., training on the ETH-UCY and our diagnostic datasets jointly:
```
python train.py --exp-id <output tag> --dataset s2r --reg-type baseline --dataset-path <path to the ETH-UCY dataset> --num-encoder-layers 1 --num-decoder-layers 1 --num-epochs 50 --learning-rate-sched 10 20 30 40 50 --low-data 1.0 --dataset-path-real <path to the ETH-UCY dataset> --dataset-path-synth <path to the synthetic dataset> --save-dir <directory for saving results> 
```

To run the data augmentation sim2real:
```
python train.py --exp-id <output tag> --dataset s2r --reg-type augment --dataset-path <path to the ETH-UCY dataset> --num-encoder-layers 1 --num-decoder-layers 1 --num-epochs 50 --learning-rate-sched 10 20 30 40 50 --low-data 1.0 --dataset-path-real <path to the ETH-UCY dataset> --dataset-path-synth <path to the synthetic dataset> --save-dir <directory for saving results> 
```

### Causal Sim2Real

To run our causal contrastive sim2real:
```
python train.py --exp-id <output tag> --dataset s2r --reg-type contrastive --dataset-path <path to the ETH-UCY dataset> --num-encoder-layers 1 --num-decoder-layers 1 --num-epochs 50 --learning-rate-sched 10 20 30 40 50 --low-data 1.0 --dataset-path-real <path to the ETH-UCY dataset> --dataset-path-synth <path to the synthetic dataset> --contrastive-weight <weight of contrastive loss> --save-dir <directory for saving results> 
```

To run the causal ranking sim2real:
```
python train.py --exp-id <output tag> --dataset s2r --reg-type ranking --dataset-path <path to the ETH-UCY dataset> --num-encoder-layers 1 --num-decoder-layers 1 --num-epochs 50 --learning-rate-sched 10 20 30 40 50 --low-data 1.0 --dataset-path-real <path to the ETH-UCY dataset> --dataset-path-synth <path to the synthetic dataset> --ranking-weight <weight of ranking loss> --save-dir <directory for saving results> 
```

### Evaluation

To evaluate a model:
```
python evaluate.py --dataset-path <path to the testing data> --models-path <path to the model> --dataset s2r
```

For experiments in low-data regimes, we keep the same training steps while reducing `--low-data` and `--learning-rate-sched` accordingly. For example, we set `--low-data` to 0.5 and `--learning-rate-sched` to 20 40 60 80 100, for experiments with 50% data.

## Results

Comparison of different methods in terms of causal awareness:
<p align="left">
  <img src="docs/ace.png" height="200">
</p>

Comparison of different methods in terms of out-of-distribution robustness:
<p align="left">
  <img src="docs/ood.png" height="200">
</p>

Comparison of different transfer methods from simulation to the ETH-UCY dataset:
<p align="left">
  <img src="docs/transfer.png" height="200">
</p>

## Citation

If you find this work useful in your research, please consider cite:

```
@article{socialcausality2023,
  title={What If You Were Not There? Learning Causally-Aware Representations of Multi-Agent Interactions},
  journal={openreview},
  year={2023}
}
```
