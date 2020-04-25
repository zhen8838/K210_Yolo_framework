CFG = config/default.yml
CKPT = None
IMG = None
RES = None
PY = None

all:
	@echo please use \"make train\" or other ...

jit_train:
	export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" && python3 ./keras_train.py --config ${CFG}

jit_eager_train:
	export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" && python3 ./eager_train.py --config ${CFG}

eager_train:
	python3 ./eager_train.py --config ${CFG}

train_gan:
	export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" && python3 ./eager_train_gan.py --config ${CFG}

infer_gan:
	python3 ./eager_infer_gan.py ${CKPT} ${IMG} --results_path ${RES}

train:
	python3 ./keras_train.py --config ${CFG}

infer:
	python3 ./keras_inference.py ${CKPT} ${IMG} --results_path ${RES}
	
eval:
	python3 ./keras_eval.py ${CKPT}
	
time_profile:
	kernprof -l ${PY} && python -m line_profiler $(notdir ${PY}).lprof
	