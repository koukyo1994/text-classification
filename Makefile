IMAGETAG := "workspace:1.1"
EXP := "exp"
NTRIAL := 200
NJOBS := 3
EMB := $$HOME/Data/Embeddings/model.vec
DEVICE := "cpu"
NEPOCHS := 10

docker/build:
	docker build -t ${IMAGETAG} .

bash:
	docker run -v `pwd`:/app -it ${IMAGETAG} /bin/bash

download:
	make -C input

unittest:
	docker run -v `pwd`:/app -it ${IMAGETAG} python -m unittest discover test/

train/lr:
	docker run -v `pwd`:/app -it ${IMAGETAG} python lr_benchmark.py --exp ${EXP}

train/lightgbm:
	docker run -v `pwd`:/app -it ${IMAGETAG} python lightgbm_benchmark.py --exp ${EXP}

optuna/lr:
	docker run -v `pwd`:/app -it ${IMAGETAG} python optuna_lr.py --exp ${EXP} --ntrial ${NTRIAL} --n_jobs ${NJOBS}

optuna/lightgbm:
	docker run -v `pwd`:/app -it ${IMAGETAG} python optuna_lightgbm.py --exp ${EXP} --ntrial ${NTRIAL} --n_jobs ${NJOBS}

train/lstm:
	python lstm_benchmark.py --exp ${EXP} --embedding ${EMB} --device ${DEVICE} --n_epochs ${NEPOCHS}
