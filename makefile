PYTHON=python
path=examples
recursive=True

make:
	@echo Installing pystokes...
	${PYTHON} setup.py install build_ext --inplace

clean-local:
	@echo removing local compiled files
	rm pystokes/*.c pystokes/*.html pystokes/*.cpp

clean:
	@echo removing all compiled files
	${PYTHON} setup.py clean
	rm pystokes/*.c pystokes/*.html pystokes/*so
	
env:
	@echo creating conda environment...
	conda env create --file environment.yml
	# conda activate pystokes
	@echo use make to install pystokes

test:
	@echo testing pystokes...
	cd pystokes && python installTests.py

nbtest:
	@echo testing example notebooks...
	@echo test $(path)
	cd examples && python testNotebooks.py --path $(path) --recursive $(recursive)
