#!/usr/bin/env sh
DATA=/media/red/DATA/material/BOT/BOT-origin
MY=/media/red/DATA/data/BOT

echo "Creating train lmdb.."
rm -rf $MY/ccrvc_train_lmdb
build/tools/convert_imageset \
--shuffle \
--resize_height=224 --resize_width=224 \
$DATA/ $DATA/train.txt $MY/ccrvc_train_lmdb

echo "Creating test lmdb.."
rm -rf $MY/ccrvc_test_lmdb
build/tools/convert_imageset \
--shuffle \
--resize_width=224 \
--resize_height=224 \
$DATA/ $DATA/val.txt $MY/ccrvc_test_lmdb

echo " Done..."
