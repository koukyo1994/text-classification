IMAGETAG = "workspace:1.1"
EXP = "exp"

docker/build:
	docker build -t ${IMAGETAG} .

bash:
	docker run -v `pwd`:/app -it ${IMAGETAG} /bin/bash

download:
	make -C input

unittest:
	docker run -v `pwd`:/app -it ${IMAGETAG} python -m unittest discover test/

train:
	docker run -v `pwd`:/app -it ${IMAGETAG} python lr_benchmark.py --exp ${EXP}

optuna/lightgbm:
	docker run -v `pwd`:/app -it ${IMAGETAG} python optuna_lightgbm.py --exp ${EXP}
