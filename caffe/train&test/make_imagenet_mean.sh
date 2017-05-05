#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/media/red/DATA/data/BOT/
DATA=/media/red/BRIDGE/data/BOT/
TOOLS=/root/caffe-inland/build/tools

$TOOLS/compute_image_mean $EXAMPLE/ccrvc_train_lmdb \
  $DATA/mean.binaryproto

echo "Done."
