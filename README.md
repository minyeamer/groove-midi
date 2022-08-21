# Groove MIDI
- POZAlabs task

## Prerequisites

- make conda environment
```bash
$ make env                  # create anaconda environment
$ conda activate <new_env>  # activate anaconda environment
$ make setup                # initial setup for the project
# add jupyter notebook kernel
$ python -m ipykernel install --user --name <new_env> --display-name <display_name> 
```

## How to Use

### Train

```bash
python3 run.py \
  --config=hierdec-drums_4bar_small \
  --config_file=data/config.json \
  --run_dir=saved/checkpoints/drums_4bar \
  --num_steps=10 \
  --mode=train
```

### Sample

```bash
python3 run.py \
  --config=hierdec-drums_4bar_small \
  --config_file=data/config.json \
  --checkpoint_file=saved/checkpoints/drums_4bar/train/model.ckpt-10 \
  --output_dir=generated/test \
  --num_outputs=5 \
  --mode=sample
```
