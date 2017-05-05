time ./build/tools/caffe train -solver models/bvlc_googlenet/quick_quick_solver.prototxt \
-snapshot=/media/red/BRIDGE/models/BOT/quick_quick_iter_220000.solverstate -gpu 0 \
2>&1 |tee models/bvlc_googlenet/100_200w_log/out.log

