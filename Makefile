.PHONY : doc

all: install

install: clean_c
	pip3 install --user -e .

clean: clean_pyc clean_c clean_doc

ext: clean_c
	cd evaltools/scores && python3 setup.py build_ext --inplace

doc:
	cd doc && $(MAKE) html

test:
	cd tests && python3 chart_catalog.py

tarball:
	@version=`python3 -c "import evaltools;print(evaltools.__version__);"`; \
	output_dir=evaltools_v$${version}; \
	echo $${version}; \
	mkdir $${output_dir}; \
	mkdir -p $${output_dir}/evaltools; \
	mkdir -p $${output_dir}/evaltools/scores; \
	mkdir -p $${output_dir}/evaltools/plotting; \
	mkdir -p $${output_dir}/doc/source; \
	mkdir -p $${output_dir}/doc/sample_data/observations; \
	mkdir -p $${output_dir}/doc/sample_data/MFMforecast/J0; \
	mkdir -p $${output_dir}/doc/sample_data/MFMforecast/J1; \
	mkdir -p $${output_dir}/doc/sample_data/MFMforecast/J2; \
	mkdir -p $${output_dir}/doc/sample_data/MFMforecast/J3; \
	mkdir -p $${output_dir}/doc/sample_data/ENSforecast/J0; \
	mkdir -p $${output_dir}/doc/sample_data/ENSforecast/J1; \
	mkdir -p $${output_dir}/doc/sample_data/ENSforecast/J2; \
	mkdir -p $${output_dir}/doc/sample_data/ENSforecast/J3; \
	mkdir -p $${output_dir}/tests; \
	for fic in `git ls-tree -r develop  --name-only`; do \
		cp -f $${fic} $${output_dir}/$${fic}; \
	done; \
	rm -f $${output_dir}/.gitignore; \
	cd $${output_dir}; \
	zip -r evaltools_v$${version}.zip *; \
	mv evaltools_v$${version}.zip ../;

clean_pyc:
	-find . -name '*.py[co]' -exec rm {} \;

clean_c:
	-rm evaltools/scores/_fastimpl.c

clean_doc:
	-rm -r doc/_build
