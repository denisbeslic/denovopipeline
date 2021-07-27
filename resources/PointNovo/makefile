clean:
	rm -rf build
	rm -f deepnovo_cython_modules.c
	rm -f deepnovo_cython_modules*.so

.PHONY: build
build: clean
	python deepnovo_cython_setup.py build_ext --inplace

.PHONY: train
train:
	python main.py --train

.PHONY: valid
valid:
	python main.py --valid

.PHONY: denovo
denovo:
	python main.py --search_denovo

.PHONY: test
test:
	python main.py --test

.PHONY: db
db:
	python main.py --search_db
