export CUDA_VISIBLE_DEVICES=0
python ./train.py configs/temporalmaxer_h2o_slowfast.yaml --save_ckpt_dir "./ckpt/h2o_slowfast"