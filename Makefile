clean:
	rm -rf __pycache__ sources/__pycache__ thetas.json

install:
	python -r requirements.txt

train:
	@python train.py --plot

predict:
	@python predict.py