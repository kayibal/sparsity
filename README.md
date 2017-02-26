# Sparsity
Sparse data processing toolbox. It builds on top of pandas and scipy to provide DataFrame
like API to work with sparse categorical data. 

It also provides a extremly fast C level 
interface to read from traildb databases. This make it a highly performant package to use
for dataprocessing jobs especially such as log processing and/or clickstream ot click through data. 

In combination with dask it provides support to execute complex operations on 
a concurrent/distributed level.

# Motivation
Many tasks especially in data analytics and machine learning domain make use of sparse
data structures to support the input of high dimensional data. 

This project was started
to build an efficient homogen sparse data processing pipeline. As of today dask has no
support for something as an sparse dataframe. We process big amounts of highdimensional data
on a daily basis at [datarevenue](http://datarevenue.com) and our favourite language 
and ETL framework are python and dask. After chaining many function calls on scipy.sparse 
csr matrices that involved handling of indices and column names to produce a sparse data
pipeline I decided to start this project.

This package might be especially usefull to you if you have very big amounts of 
sparse data such as clickstream data, categorical timeseries, log data or similarly sparse data.

# Traildb access?
[Traildb](http://traildb.io/) is an amazing log style database. It was released recently 
by AdRoll. It compresses event like data extremly efficient. Furthermore it provides a 
fast C-level api to query it. 

Traildb has also python bindings but you still might need to iterate over many million 
of users/trail or even both which has quite some overhead in python. 
Therefore sparsity provides high speed access to the database in form of SparseFrame objects. 
These are fast, efficient and intuitive enough to do further processing on. 

*ATM uuid and timestamp informations are lost but they will be provided as a pandas.MultiIndex 
handled by the SparseFrame in a (very soon) future release.*

````
In [1]: from sparsity import SparseFrame

In [2]: sdf = SparseFrame.read_traildb('pydata.tdb', field="title")

In [3]: sdf.head()
Out[3]: 
   0      1      2      3      4      ...    37388  37389  37390  37391  37392
0    1.0    0.0    0.0    0.0    0.0  ...      0.0    0.0    0.0    0.0    0.0
1    1.0    0.0    0.0    0.0    0.0  ...      0.0    0.0    0.0    0.0    0.0
2    1.0    0.0    0.0    0.0    0.0  ...      0.0    0.0    0.0    0.0    0.0
3    1.0    0.0    0.0    0.0    0.0  ...      0.0    0.0    0.0    0.0    0.0
4    1.0    0.0    0.0    0.0    0.0  ...      0.0    0.0    0.0    0.0    0.0

[5 rows x 37393 columns]

In [6]: %%timeit
   ...: sdf = SparseFrame.read_traildb("/Users/kayibal/Code/traildb_to_sparse/traildb_to_sparse/traildb_to_sparse/sparsity/test/pydata.tdb", field="title")
   ...: 
10 loops, best of 3: 73.8 ms per loop

In [4]: sdf.shape
Out[4]: (109626, 37393)
````

# But wait pandas has SparseDataFrames and SparseSeries
Pandas has it's own implementation of sparse datastructures. Unfortuantely this structures
performs quite badly with a groupby sum aggregation which we also often use. Furthermore
 doing a groupby on a pandasSparseDataFrame returns a dense DataFrame. This makes chaining
  many groupby operations over multiple files cumbersome and less efficient. Consider 
following example:

```
In [1]: import sparsity
   ...: import pandas as pd
   ...: import numpy as np
   ...: 

In [2]: data = np.random.random(size=(1000,10))
   ...: data[data < 0.95] = 0
   ...: uids = np.random.randint(0,100,1000)
   ...: combined_data = np.hstack([uids.reshape(-1,1),data])
   ...: columns = ['id'] + list(map(str, range(10)))
   ...: 
   ...: sdf = pd.SparseDataFrame(combined_data, columns = columns, default_fill_value=0)
   ...: 

In [3]: %%timeit
   ...: sdf.groupby('id').sum()
   ...: 
1 loop, best of 3: 462 ms per loop

In [4]: res = sdf.groupby('id').sum()
   ...: res.values.nbytes
   ...: 
Out[4]: 7920

In [5]: data = np.random.random(size=(1000,10))
   ...: data[data < 0.95] = 0
   ...: uids = np.random.randint(0,100,1000)
   ...: sdf = sparsity.SparseFrame(data, columns=np.asarray(list(map(str, range(10)))), index=uids)
   ...: 

In [6]: %%timeit
   ...: sdf.groupby_sum()
   ...: 
The slowest run took 4.20 times longer than the fastest. This could mean that an intermediate result is being cached.
1000 loops, best of 3: 1.25 ms per loop

In [7]: res = sdf.groupby_sum()
   ...: res.__sizeof__()
   ...: 
Out[7]: 6128
```

I'm not quite sure if there is some cached result but I don't think so. This only uses a 
smart csr matrix multiplication to do the operation.