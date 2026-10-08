CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc-per-node=4 \
  docs/tutorial/cli/generation/train.py
