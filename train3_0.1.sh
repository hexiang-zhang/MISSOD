
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_cefpn_001.py  --num-gpus 4   --config /mnt/sdd/zhanghexiang/unbiased-teacher/configs/buchong/config_cefpn_001.yaml SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 >cefpnfull.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_pan_001.py  --num-gpus 4   --config /mnt/sdd/zhanghexiang/unbiased-teacher/configs/buchong/config_panet_001.yaml SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 >panfull.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_hr_001.py  --num-gpus 4   --config /mnt/sdd/zhanghexiang/unbiased-teacher/configs/buchong/config_hr_001.yaml SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 >hrfull.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net_mhfpn_001.py  --num-gpus 4   --config /mnt/sdd/zhanghexiang/unbiased-teacher/configs/buchong/config_mhfpn2_001.yaml SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 >mhfpnfull.log 2>&1
