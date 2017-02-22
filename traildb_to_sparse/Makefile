test: sparsity/traildb.cpython-35m-darwin.so
	py.test sparsity/test

sparsity/traildb.cpython-35m-darwin.so: sparsity/traildb.pyx
	python setup.py build_ext --inplace

clean:
	rm sparsity/traildb.c sparsity/traildb.cpython-35m-darwin.so