MetaNeighbor: a method to rapidly assess cell type identity using both functional and random gene sets
================
MetaNeighbor allows users to quantify cell type replicability across datasets using neighbor voting.

Please refer the [notebooks](./notebooks) for tutorials on how to use MetaNeighbor and consider citing [Crow et al (2018) Nature Communications](https://www.nature.com/articles/s41467-018-03282-0) if you find MetaNeighbor useful in your research.

The code was translated from R to Python, you can find the R package in this (repository)[https://github.com/gillislab/MetaNeighbor/]


## Installation

pyMN depends on mostly standard numerical and single cell RNAseq analysis python packages.
 * Numpy
 * Scipy>=1.4
 * Pandas>=.21
 * networkx
 * anndata>=.7
 * scanpy
 * matplotlib
 * seaborn
 * upsetplot

The two less standard packages are pygraphviz and bottleneck

### pygraphviz

There is currently (as of October 2020) a bug in the installation pygraphviz on pip. For mac and linux the conda installation `conda install -c anaconda pygraphviz` appears to work without issues, but for Windows you likely will need to use a distribution from this personal conda channel. `conda install -c alubbock pygraphviz`,
You can learn more about these issues [on this stackoverflow thread](https://stackoverflow.com/questions/59707234/issues-installing-pygrahviz-fatal-error-c1083-cannot-open-include-file-graph)


### bottleneck
On UNIX we haven't seen any issues with installing bottleneck using pip or conda, but on Windows we have found that, without already having Microsoft Visual C++ Build Tools already installed on your system, the easiest way to install bottleneck is with `conda install -c anaconda bottleneck`. If you have MVCBT then it should work with pip when you install pyMN. 


### Installing pyMN
You can either install this by cloning the repository, moving into it and then running:
  `python setup.py install` 

  or
   
  `pip install .` 

If you don't want to clone the repository you can just run:
	`pip install git+https://github.com/gillislab/pyMN#egg=pymetaneighbor`


## Usage

For detailed tutorials refer to the [notebooks](./notebooks) folder for 3 different protocols. In general the code works similar to other packages that heavily interact with Anndata objects (like scanpy and the packages in scanpy.external). You pass an AnnData object and the parameters necesssary to run the function as you intend it to. There are 2 types of parameters.
	1) Ones that describe data already in the Anndata
	2) Ones that tune the functionality of the method

### Memory-constrained sparse workflows

For sparse inputs that do not fit the legacy dense fast path, highly variable
gene selection and de novo `MetaNeighborUS(fast_version=True)` can be run in
bounded batches:

```python
pymn.variableGenes(
    adata,
    study_col="donor",
    memory_constrained=True,
    gene_batch_size=1024,
    n_jobs=8,
)

pymn.MetaNeighborUS(
    adata,
    study_col="donor",
    ct_col="class",
    fast_version=True,
    memory_constrained=True,
    cell_batch_size=256,
    centroid_n_jobs=4,
    score_batch_size=64,
    temporary_directory="/path/to/fast/local/scratch",
)
```

When each study is stored in a separate H5AD, it can be processed directly
from disk. The files are first opened in read-only backed mode to identify the
genes shared by every study. Only those shared genes are then processed, one
all-cell gene batch at a time, and worker threads divide the in-memory batch:

```python
hvgs = pymn.variableGenesFromH5ADs(
    ["study_1.h5ad", "study_2.h5ad"],
    gene_batch_size=1024,
    n_jobs=8,
)
```

The returned list contains genes selected as highly variable in every file,
in the gene order of the first H5AD. Backed file handles remain open while
expression data are processed sequentially, but only the current shared-gene
batch is loaded into memory.

`cell_batch_size` bounds each dense rank-normalization and vote-generation
batch. `centroid_n_jobs` processes independent centroid cell batches with
worker threads; peak centroid-building work memory grows approximately as
`cell_batch_size * centroid_n_jobs`, while the per-worker batch size remains
unchanged. `centroid_n_jobs` does not multiply or otherwise change
`score_batch_size`.
`score_batch_size` bounds the vote columns ranked in memory for AUROC
calculation. Votes are stored in a temporary memory-mapped file for one test
study at a time and deleted after that study. The largest temporary file is
approximately `largest_study_cell_count * study_cell_type_count * 8` bytes with
the default `vote_dtype="float64"`. `vote_dtype="float32"` halves centroid and
temporary-vote storage but can introduce small numerical differences.

The memory-constrained path accepts categorical study and cell-type columns.
It currently applies to de novo runs with `fast_version=True`; pretrained and
non-fast runs continue to use the legacy implementations.

When enabled, the memory-constrained functions print their batch and HVG worker
settings, detected numerical thread counts, estimated working and temporary-file
sizes, per-study progress, temporary-file cleanup, and completion time.

The reproducible three-core HMBA benchmark, saved correctness matrices, and
comparison command are documented in [benchmarks/README.md](./benchmarks/README.md).


## Issues and Bugs

If you have any issues or find any bugs please use the [Issue Tracker](https://github.com/gillislab/pyMN/issues)
