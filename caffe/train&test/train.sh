time ./build/tools/caffe train -solver models/bvlc_googlenet/quick_quick_solver.prototxt -gpu 1 \
2>&1 |tee models/bvlc_googlenet/0_20w_log/out.log

