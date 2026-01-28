#!/usr/bin/env bash
echo "Running inference on" ${1}
# --val_dataset synthia bdd100k cityscapes mapillary gtav \
     python -m torch.distributed.launch --nproc_per_node=1 valid.py \
        --val_dataset cityscapes bdd100k synthia mapillary \
        --arch network.deepv3.DeepR50V3PlusD \
        --wt_layer 0 0 4 4 4 0 0 \
        --date 0101 \
        --exp r50os16_gtav_ibn \
        --snapshot ${1}
