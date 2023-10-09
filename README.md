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

## Baselines

To train the [AutoBots](https://openreview.net/forum?id=Dup_dDqkZC5) baseline:
```
python train.py --exp-id baseline --save-dir <results directory, e.g., ./ > --dataset-path <path to synth dataset> --evaluate_causal
```

To run the [data augmentation](https://arxiv.org/abs/2207.03586) baseline:
```
python train.py --exp-id baseline --save-dir <results directory, e.g., ./> --dataset-path <path to synth dataset> --evaluate_causal --reg-type augment
```

## Regularization

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

To evaluate on OOD sets:
```
python evaluate.py --models-path <path to the model> --dataset-path <path to the ood dataset>
```

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
