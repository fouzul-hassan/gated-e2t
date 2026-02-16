python test.py
python predict.py --checkpoint ./runs/dev-dist/wandb/latest-run/files-checkpoints/epoch=199-step=397600.ckpt --task all --use_llm

python train_with_jepa_encoder_bart.py
