docker run -it \
    --shm-size 11G \
    --gpus all  \
    -v $(pwd):/temporalmaxer \
    -v /home/fai/Paul/paul_dataset/:/temporalmaxer/paul_dataset/ \
    paul/temporalmaxer:1.1 \
    bash