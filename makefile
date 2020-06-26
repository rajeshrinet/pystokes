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
	cd tests && python shortTests.py

nbtest:
	@echo testing example notebooks...
	@echo test $(path)
	cd tests && python notebookTests.py --path $(path) --recursive $(recursive)

pypitest:
	@echo testing pystokes...
	python setup.py sdist bdist_wheel
	python -m twine upload --repository testpypi dist/*

pypi:
	@echo testing pystokes...
	python setup.py sdist bdist_wheel	
	python -m twine upload dist/*

cycov:
	python setup.py build_ext --force --inplace --define CYTHON_TRACE
	pytest tests/shortTests.py  --cov=./ --cov-report=xml
