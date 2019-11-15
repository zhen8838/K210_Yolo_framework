CFG = config/default.yml
CKPT = None
IMG = None
RES = None
all:
	@echo please use \"make train\" or other ...

jit_train:
	export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" && python3 ./keras_train.py --config ${CFG}

train:
	python3 ./keras_train.py --config ${CFG}

infer:
	python3 ./keras_inference.py ${CKPT} ${IMG} --results_path ${RES}
	
eval:
	python3 ./keras_eval.py ${CKPT}