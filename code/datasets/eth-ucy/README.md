# ETH-UCY Setup
Extract the data :

```
python create_data_npys.py --raw-dataset-path /path/to/synth_data/ --output-npy-path /path/to/output_npys --split <split: train, test, or val>
```

This script will generate preprocessed data into a numpy file called `{split}_{name of dataset}.npy`.
