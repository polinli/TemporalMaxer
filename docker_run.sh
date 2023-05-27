docker run -it \
    --shm-size 11G \
    --gpus all  \
    -v $(pwd):/temporalmaxer \
    -v ~/paul/dataset/:/temporalmaxer/data/ \
    -v ~/paul/dataset/hdd_data/:/temporalmaxer/data/hdd_data \
    paul/temporalmaxer:latest \
    bash
