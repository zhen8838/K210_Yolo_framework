CKPT=""
PB=Freeze_save
IAA=False
ILR=0.0005
CLSNUM=20
BATCH=32
DATASET=voc
MAXEP=10
MODEL=yolo_mobilev1
DEPTHMUL=0.75
MODELCMP=~/Documents/kendryte-model-compiler
LRDECAYFACTOR=0
OBJWEIGHT=1
NOOBJWEIGHT=1
WHWEIGHT=1
IMG=data/people.jpg
SPLITFACTOR=0.05
OBJTHRESH=0.7
IOUTHRESH=0.5
PRUNE=False
INITSPARSITY=0.5
FINALSPARSITY=0.9
END_EPOCH=5
FREQUENCY=100
IMGSIZE=224 320
OUTSIZE=7 10 14 20
ANCNUM=3
LOW=0.0 0.0
HIGH=1.0 1.0

all:
	@echo please use \"make train\" or other ...

train:
	python3 ./keras_train.py \
			--train_set ${DATASET} \
			--class_num ${CLSNUM} \
			--pre_ckpt ${CKPT} \
			--model_def ${MODEL} \
			--depth_multiplier ${DEPTHMUL} \
			--augmenter ${IAA} \
			--image_size ${IMGSIZE} \
			--output_size ${OUTSIZE} \
			--batch_size ${BATCH} \
			--rand_seed 3 \
			--max_nrof_epochs ${MAXEP} \
			--init_learning_rate ${ILR} \
			--learning_rate_decay_factor ${LRDECAYFACTOR} \
			--obj_weight ${OBJWEIGHT} \
			--noobj_weight ${NOOBJWEIGHT} \
			--wh_weight ${WHWEIGHT} \
			--obj_thresh ${OBJTHRESH} \
			--iou_thresh ${IOUTHRESH} \
			--vaildation_split ${SPLITFACTOR} \
			--log_dir log \
			--is_prune ${PRUNE} \
			--prune_initial_sparsity ${INITSPARSITY} \
			--prune_final_sparsity ${FINALSPARSITY} \
			--prune_end_epoch ${END_EPOCH} \
			--prune_frequency ${FREQUENCY} 

freeze:
	python3 ./keras_freeze.py ${CKPT}
			
inference:
	python3	./keras_inference.py \
			${CKPT} \
			${IMG} \
			--train_set ${DATASET} \
			--class_num ${CLSNUM} \
			--model_def ${MODEL} \
			--depth_multiplier ${DEPTHMUL} \
			--obj_thresh ${OBJTHRESH} \
			--iou_thresh ${IOUTHRESH} \
			--image_size ${IMGSIZE} \
			--output_size ${OUTSIZE}

anchors:
	python3 ./make_anchor_list.py \
			${DATASET} \
			--max_iters 10 \
			--is_random True \
			--in_hw ${IMGSIZE} \
			--out_hw ${OUTSIZE} \
			--anchor_num ${ANCNUM} \
			--low ${LOW} \
			--high ${HIGH}

build_kfpkg:
	cd ~/workspace/kendryte-standalone-sdk-0.5.6/build && make && zip -r kpu_yolov3.kfpkg  flash-list.json kpu_yolov3.bin yolo.kmodel && cd -
	
download:
	python /home/zqh/Documents/kflash.py/kflash.py ~/workspace/kendryte-standalone-sdk-0.5.6/build/kpu_yolov3.kfpkg -B kd233 -p /dev/ttyUSB0 -b 2000000 -t