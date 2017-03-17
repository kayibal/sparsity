test: sparsity/traildb.cpython-35m-darwin.so
	py.test sparsity/test -s

sparsity/traildb.cpython-35m-darwin.so: sparsity/_traildb.pyx
	python setup.py build_ext --inplace

clean:
	rm -f sparsity/_traildb.c sparsity/_traildb.cpython-35m-darwin.so