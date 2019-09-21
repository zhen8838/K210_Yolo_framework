CFG = config/default.yml

all:
	@echo please use \"make train\" or other ...

train:
	export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
	python3 ./keras_train.py --config ${CFG}