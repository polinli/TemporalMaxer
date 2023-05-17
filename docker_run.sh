docker run -it \
    --shm-size 11G \
    --gpus all  \
    -v $(pwd):/temp_maxer \
    -v /home/fai/Paul/paul_dataset/:/temp_maxer/paul_dataset/ \
    paul/temporal-maxer:latest \
    bash