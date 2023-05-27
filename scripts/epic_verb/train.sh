export CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=1 python ./train.py configs/temporalmaxer_epic_slowfast_verb.yaml --save_ckpt_dir "./ckpt/verb_epic"
