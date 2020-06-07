PYTHON=python
path=examples
recursive=True

make:
	@echo Installing pystokes...
	${PYTHON} setup.py install

clean-local:
	@echo removing local compiled files
	rm pystokes/*.c pystokes/*.html pystokes/*.cpp

clean:
	@echo removing all compiled files
	${PYTHON} setup.py clean
	rm pystokes/*.c pystokes/*.html

env:
	@echo creating conda environment...
	conda env create --file environment.yml
	# conda activate pystokes
	@echo use make to install pystokes

test:
	@echo testing pystokes...
	cd tests && python installTests.py


pypitest:
	@echo testing pystokes...
	python -m twine upload --repository testpypi dist/*

pypi:
	@echo testing pystokes...
	python -m twine upload dist/*


nbtest:
	@echo testing example notebooks...
	@echo test $(path)
	cd tests && python testNotebooks.py --path $(path) --recursive $(recursive)
