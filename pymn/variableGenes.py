import numpy as np
import bottleneck
from scipy import sparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import gc

from .utils import format_bytes, numerical_thread_summary


def _select_genes_from_statistics(median, variance):
    """Select genes from per-gene median and variance vectors."""
    quantiles = np.linspace(0, 1, 11)
    try:
        # NumPy >= 1.22 renamed ``interpolation`` to ``method`` and recent
        # releases no longer accept the old keyword.
        bins = np.quantile(median, q=quantiles, method="midpoint")
    except TypeError:
        # NumPy 1.21 and older accept only ``interpolation``.
        bins = np.quantile(median,
                           q=quantiles,
                           interpolation="midpoint")
    digits = np.digitize(median, bins, right=True)

    selected_genes = np.zeros_like(digits)
    for i in np.unique(digits):
        filt = digits == i
        var_tmp = variance[filt]
        bins_tmp = np.nanquantile(var_tmp, q=np.linspace(0, 1, 5))
        g = np.digitize(var_tmp, bins_tmp)
        selected_genes[filt] = (g >= 4).astype(float)
    return selected_genes.astype(bool)


def compute_var_genes(adata, return_vect=True):
    """Compute variable genes for an indiviudal dataset


    Arguments:
        adata {[type]} -- AnnData object containing a signle dataset

    Keyword Arguments:
        return_vect {bool} -- Boolean to store as adata.var['higly_variance']
            or return vector of booleans for varianble gene membership (default: {False})

    Returns:
        np.ndarray -- None if saving in adata.var['highly_variable'], array of booleans if returning of length ngenes
    """

    if sparse.issparse(adata.X):
        median = csc_median_axis_0(sparse.csc_matrix(adata.X))
        variance = sparse_var_axis0(adata.X)
    else:
        median = bottleneck.median(adata.X, axis=0)
        variance = np.var(adata.X,axis=0)
    selected_genes = _select_genes_from_statistics(median, variance)

    if return_vect:
        return selected_genes
    else:
        adata.var["highly_variable"] = selected_genes


def _compute_gene_statistics_block(X, row_indices, start, stop):
    """Compute median and variance for one independent gene block."""
    if sparse.issparse(X):
        block = X[row_indices, start:stop]
        median = csc_median_axis_0(sparse.csc_matrix(block))
        variance = sparse_var_axis0(block)
    else:
        block = np.asarray(X[np.ix_(row_indices, np.arange(start, stop))])
        median = bottleneck.median(block, axis=0)
        variance = np.var(block, axis=0)
    return start, stop, median, variance


def compute_var_genes_batched(X,
                              row_indices,
                              gene_batch_size=1024,
                              n_jobs=1):
    """Compute variable genes while bounding temporary data by gene batches.

    Unlike ``compute_var_genes``, this helper never constructs an AnnData view
    containing every gene for a study. Sparse row subsets are materialized only
    for the current gene batch.
    """
    if not isinstance(gene_batch_size, (int, np.integer)):
        raise TypeError("gene_batch_size must be an integer")
    if gene_batch_size <= 0:
        raise ValueError("gene_batch_size must be a positive integer")
    if not isinstance(n_jobs, (int, np.integer)):
        raise TypeError("n_jobs must be an integer")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer")

    row_indices = np.asarray(row_indices, dtype=np.intp)
    n_genes = X.shape[1]
    median = np.empty(n_genes, dtype=float)
    variance = np.empty(n_genes, dtype=float)

    blocks = [
        (start, min(start + gene_batch_size, n_genes))
        for start in range(0, n_genes, gene_batch_size)
    ]

    if n_jobs == 1 or len(blocks) == 1:
        completed_blocks = (
            _compute_gene_statistics_block(X, row_indices, start, stop)
            for start, stop in blocks
        )
        for start, stop, block_median, block_variance in completed_blocks:
            median[start:stop] = block_median
            variance[start:stop] = block_variance
    else:
        worker_count = min(n_jobs, len(blocks))
        with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="pymn-hvg") as executor:
            futures = [
                executor.submit(
                    _compute_gene_statistics_block,
                    X,
                    row_indices,
                    start,
                    stop,
                )
                for start, stop in blocks
            ]
            for future in as_completed(futures):
                start, stop, block_median, block_variance = future.result()
                median[start:stop] = block_median
                variance[start:stop] = block_variance

    return _select_genes_from_statistics(median, variance)


def _validate_batch_parameters(gene_batch_size, n_jobs):
    """Validate the batching parameters shared by the HVG entry points."""
    if not isinstance(gene_batch_size, (int, np.integer)):
        raise TypeError("gene_batch_size must be an integer")
    if gene_batch_size <= 0:
        raise ValueError("gene_batch_size must be a positive integer")
    if not isinstance(n_jobs, (int, np.integer)):
        raise TypeError("n_jobs must be an integer")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer")


def _compute_matrix_gene_statistics(X):
    """Compute median and variance for every column of an in-memory matrix."""
    if sparse.issparse(X):
        median = csc_median_axis_0(sparse.csc_matrix(X))
        variance = sparse_var_axis0(X)
    else:
        X = np.asarray(X)
        median = bottleneck.median(X, axis=0)
        variance = np.var(X, axis=0)
    return np.asarray(median), np.asarray(variance)


def _compute_loaded_statistics_block(X, start, stop):
    """Compute statistics for a column range of an in-memory gene batch."""
    block_median, block_variance = _compute_matrix_gene_statistics(
        X[:, start:stop]
    )
    return start, stop, block_median, block_variance


def _compute_loaded_batch_statistics(X, n_jobs, executor=None):
    """Compute one loaded gene batch, splitting its columns across threads."""
    n_genes = X.shape[1]
    worker_count = min(n_jobs, n_genes)
    block_size = int(np.ceil(n_genes / worker_count))
    blocks = [
        (start, min(start + block_size, n_genes))
        for start in range(0, n_genes, block_size)
    ]
    median = np.empty(n_genes, dtype=float)
    variance = np.empty(n_genes, dtype=float)

    if worker_count == 1:
        completed_blocks = (
            _compute_loaded_statistics_block(X, start, stop)
            for start, stop in blocks
        )
    else:
        futures = [
            executor.submit(_compute_loaded_statistics_block, X, start, stop)
            for start, stop in blocks
        ]
        completed_blocks = (
            future.result() for future in as_completed(futures)
        )

    for start, stop, block_median, block_variance in completed_blocks:
        median[start:stop] = block_median
        variance[start:stop] = block_variance
    return median, variance


def _load_backed_gene_block(adata, start, stop):
    """Materialize one all-cell gene block from a backed AnnData object."""
    block = adata.X[:, start:stop]
    # Some newer AnnData backed-array implementations expose ``to_memory``;
    # h5py datasets and the older SparseDataset already materialize on slicing.
    if hasattr(block, "to_memory"):
        block = block.to_memory()
    if not sparse.issparse(block):
        block = np.asarray(block)
    return block


def _load_backed_gene_index_block(adata, gene_indices):
    """Materialize selected backed columns in the requested gene order."""
    gene_indices = np.asarray(gene_indices, dtype=np.intp)
    order = np.argsort(gene_indices)
    sorted_indices = gene_indices[order]
    block = adata.X[:, sorted_indices]
    if hasattr(block, "to_memory"):
        block = block.to_memory()
    if not sparse.issparse(block):
        block = np.asarray(block)

    # h5py requires increasing indices for fancy indexing. Restore the shared
    # gene order only after the selected columns have been materialized.
    if not np.array_equal(order, np.arange(len(order))):
        block = block[:, np.argsort(order)]
    return block


def _compute_backed_adata_gene_statistics(adata,
                                           gene_batch_size,
                                           n_jobs,
                                           verbose,
                                           gene_indices=None):
    """Stream selected all-cell gene batches from a backed AnnData object."""
    n_cells, total_genes = adata.shape
    if n_cells == 0:
        raise ValueError("H5AD files must contain at least one cell")
    if total_genes == 0:
        raise ValueError("H5AD files must contain at least one gene")

    if gene_indices is None:
        n_genes = total_genes
    else:
        gene_indices = np.asarray(gene_indices, dtype=np.intp)
        n_genes = len(gene_indices)
        if n_genes == 0:
            raise ValueError("At least one gene must be selected")

    median = np.empty(n_genes, dtype=float)
    variance = np.empty(n_genes, dtype=float)
    batch_count = int(np.ceil(n_genes / gene_batch_size))
    worker_count = min(n_jobs, gene_batch_size, n_genes)
    executor = None
    if worker_count > 1:
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="pymn-h5ad-hvg",
        )

    try:
        for batch_number, start in enumerate(
                range(0, n_genes, gene_batch_size), start=1):
            stop = min(start + gene_batch_size, n_genes)
            if verbose:
                print(
                    "[variableGenesFromH5ADs] "
                    f"gene batch {batch_number}/{batch_count}: "
                    f"{start:,}:{stop:,}",
                    flush=True,
                )
            # Only this slice performs backed I/O. Worker threads operate on
            # the resulting in-memory block, avoiding concurrent h5py access.
            if gene_indices is None:
                block = _load_backed_gene_block(adata, start, stop)
            else:
                block = _load_backed_gene_index_block(
                    adata,
                    gene_indices[start:stop],
                )
            block_median, block_variance = _compute_loaded_batch_statistics(
                block,
                n_jobs=worker_count,
                executor=executor,
            )
            median[start:stop] = block_median
            variance[start:stop] = block_variance
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return median, variance


def variableGenesFromH5ADs(h5ad_paths,
                           gene_batch_size=1024,
                           n_jobs=1,
                           verbose=True):
    """Identify HVGs across H5AD files without loading a whole file's X.

    Each input H5AD is treated as one study. All files are first opened in
    read-only backed mode to identify their shared gene universe. Expression
    data are then processed one study at a time, and at most
    ``gene_batch_size`` shared-gene columns are explicitly loaded for all cells
    at once. The loaded columns are split among ``n_jobs`` worker threads for
    median and variance calculation.

    Per-study HVGs use the same median-bin/top-variance-quartile definition as
    :func:`variableGenes`. The returned genes are highly variable in every
    input study and present in every file. Gene identity is aligned by
    ``var_names`` (column order may differ between files), and output order
    follows the first file.

    Parameters
    ----------
    h5ad_paths : path-like or iterable of path-like
        One or more H5AD files. Each file represents a study.
    gene_batch_size : int, default 1024
        Maximum number of genes read from backed ``X`` at one time. Temporary
        worker copies can add overhead within this bound.
    n_jobs : int, default 1
        Number of worker threads used to compute statistics within each loaded
        gene batch. HDF5 reads themselves remain sequential.
    verbose : bool, default True
        Print file and gene-batch progress.

    Returns
    -------
    list of str
        Gene names selected as highly variable in every input file.
    """
    _validate_batch_parameters(gene_batch_size, n_jobs)
    if not isinstance(verbose, (bool, np.bool_)):
        raise TypeError("verbose must be a boolean")

    if isinstance(h5ad_paths, (str, bytes, Path)):
        paths = [Path(h5ad_paths)]
    else:
        try:
            paths = [Path(path) for path in h5ad_paths]
        except TypeError:
            raise TypeError(
                "h5ad_paths must be a path or an iterable of paths"
            )
    if not paths:
        raise ValueError("h5ad_paths must contain at least one file")
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "H5AD file(s) do not exist: " + ", ".join(missing_paths)
        )

    # Import lazily so the existing AnnData-object API keeps its lightweight
    # import behavior.
    import anndata

    studies = []
    try:
        common_genes = None
        for study_number, path in enumerate(paths, start=1):
            if verbose:
                print(
                    "[variableGenesFromH5ADs] "
                    f"loading H5AD {study_number}/{len(paths)}: {path}",
                    flush=True,
                )
            adata = anndata.read_h5ad(str(path), backed="r")
            studies.append((path, adata))
            genes = pd.Index(adata.var_names.astype(str))
            if not genes.is_unique:
                raise ValueError(
                    f"H5AD var_names must be unique: {path}"
                )
            if adata.n_obs == 0:
                raise ValueError("H5AD files must contain at least one cell")
            if adata.n_vars == 0:
                raise ValueError("H5AD files must contain at least one gene")
            if common_genes is None:
                common_genes = genes
            else:
                common_genes = common_genes[common_genes.isin(genes)]
            if verbose:
                print(
                    "[variableGenesFromH5ADs] "
                    f"loaded H5AD {study_number}/{len(paths)}: "
                    f"cells={adata.n_obs:,}, genes={adata.n_vars:,}, "
                    f"shared_genes={len(common_genes):,}",
                    flush=True,
                )

        if len(common_genes) == 0:
            raise ValueError("Input H5AD files have no genes in common")

        highly_variable_in_all = np.ones(len(common_genes), dtype=bool)
        for study_number, (path, adata) in enumerate(studies, start=1):
            if verbose:
                print(
                    "[variableGenesFromH5ADs] "
                    f"computing variable statistics "
                    f"{study_number}/{len(paths)}: {path} "
                    f"({len(common_genes):,} shared genes)",
                    flush=True,
                )
            genes = pd.Index(adata.var_names.astype(str))
            gene_indices = genes.get_indexer(common_genes)
            if (
                len(common_genes) == adata.n_vars
                and np.array_equal(
                    gene_indices,
                    np.arange(adata.n_vars, dtype=np.intp),
                )
            ):
                gene_indices = None
            median, variance = _compute_backed_adata_gene_statistics(
                adata,
                gene_batch_size=gene_batch_size,
                n_jobs=n_jobs,
                verbose=verbose,
                gene_indices=gene_indices,
            )
            study_hvgs = _select_genes_from_statistics(median, variance)
            highly_variable_in_all &= study_hvgs

            if verbose:
                print(
                    "[variableGenesFromH5ADs] "
                    f"computed variable statistics "
                    f"{study_number}/{len(paths)}: "
                    f"selected_genes={int(study_hvgs.sum()):,}",
                    flush=True,
                )
    finally:
        for _, adata in studies:
            adata.file.close()
        # AnnData loads cell-independent annotations eagerly even in backed
        # mode, so release all handles and metadata promptly after processing.
        del studies
        gc.collect()

    result = common_genes[highly_variable_in_all].tolist()
    if verbose:
        print(
            "[variableGenesFromH5ADs] complete: "
            f"selected_genes={len(result):,}",
            flush=True,
        )
    return result


def variableGenes(adata,
                  study_col,
                  return_vect=False,
                  memory_constrained=False,
                  gene_batch_size=1024,
                  n_jobs=1):
    """Comptue variable genes across data sets

    Identifies genes with high variance compared to their median expression
    (top quartile) within each experimentCertain function

    Arguments:
        adata {AnnData} -- AnnData object containing all the single cell experiements concatenated together
        study_col {str} -- String referencing column in andata.obs that identifies study label for datasets

    Keyword Arguments:
        return_vect {bool} -- Boolean to store as adata.var['higly_variance']
            or return vector of booleans for varianble gene membership (default: {False})
        memory_constrained {bool} -- Process sparse input in gene batches instead
            of materializing a full study-by-gene sparse subset (default: {False})
        gene_batch_size {int} -- Maximum number of genes materialized per study
            when memory_constrained is True (default: {1024})
        n_jobs {int} -- Number of gene batches processed concurrently with
            worker threads when memory_constrained is True. Peak batch memory
            grows approximately linearly with this value (default: {1})

    Returns:
        np.ndarray -- None if saving in adata.var['highly_variable'], array of booleans if returning of length ngenes
    """

    assert study_col in adata.obs.columns, f"Study col '{study_col}' not in adata.obs"


    studies = np.unique(adata.obs[study_col])
    if memory_constrained:
        if not isinstance(gene_batch_size, (int, np.integer)):
            raise TypeError("gene_batch_size must be an integer")
        if gene_batch_size <= 0:
            raise ValueError("gene_batch_size must be a positive integer")
        if not isinstance(n_jobs, (int, np.integer)):
            raise TypeError("n_jobs must be an integer")
        if n_jobs <= 0:
            raise ValueError("n_jobs must be a positive integer")
        var_genes = np.ones(adata.n_vars, dtype=bool)
        study_values = adata.obs[study_col].values
        study_sizes = np.asarray([
            np.count_nonzero(study_values == study) for study in studies
        ])
        input_itemsize = np.dtype(adata.X.dtype).itemsize
        batch_genes = min(gene_batch_size, adata.n_vars)
        gene_batch_count = int(np.ceil(adata.n_vars / gene_batch_size))
        worker_count = min(n_jobs, gene_batch_count)
        dense_value_batch = (
            int(study_sizes.max()) * batch_genes * input_itemsize
        )
        print(
            "[variableGenes memory_constrained] setup\n"
            f"  cells={adata.n_obs:,}, genes={adata.n_vars:,}, "
            f"studies={len(studies):,}\n"
            f"  gene_batch_size={gene_batch_size:,}, "
            f"n_jobs={n_jobs:,}, active_workers={worker_count:,}, "
            f"largest_study_cells={int(study_sizes.max()):,}\n"
            "  estimated_dense_value_batch="
            f"{format_bytes(dense_value_batch)} "
            "(sparse temporary storage depends on nnz)\n"
            "  estimated_concurrent_dense_value_batches="
            f"{format_bytes(dense_value_batch * worker_count)}\n"
            f"  cores: {numerical_thread_summary()}",
            flush=True,
        )
        for study_number, study in enumerate(studies, start=1):
            row_indices = np.flatnonzero(study_values == study)
            print(
                "[variableGenes memory_constrained] "
                f"study {study_number}/{len(studies)}: {study!s} "
                f"({len(row_indices):,} cells)",
                flush=True,
            )
            genes_vec = compute_var_genes_batched(
                adata.X,
                row_indices,
                gene_batch_size=gene_batch_size,
                n_jobs=n_jobs,
            )
            var_genes &= genes_vec
        print(
            "[variableGenes memory_constrained] complete: "
            f"selected_genes={int(var_genes.sum()):,}",
            flush=True,
        )
    else:
        genes = adata.var_names
        var_genes_mat = pd.DataFrame(index=genes)

        for study in studies:
            slicer = adata.obs[study_col] == study
            genes_vec = compute_var_genes(adata[slicer])
            var_genes_mat.loc[:, study] = genes_vec.astype(bool)
        var_genes = np.all(var_genes_mat, axis=1)
    if return_vect:
        return var_genes
    else:
        adata.var["highly_variable"] = var_genes


def _get_elem_at_rank(rank, data, n_negative, n_zeros):
    """Find the value in data augmented with n_zeros for the given rank"""
    if rank < n_negative:
        return data[rank]
    if rank - n_negative < n_zeros:
        return 0
    return data[rank - n_zeros]


def _get_median(data, n_zeros):
    """Compute the median of data with n_zeros additional zeros.
    This function is used to support sparse matrices; it modifies data in-place
    """
    n_elems = len(data) + n_zeros
    if not n_elems:
        return np.nan
    n_negative = np.count_nonzero(data < 0)
    middle, is_odd = divmod(n_elems, 2)
    data.sort()

    if is_odd:
        return _get_elem_at_rank(middle, data, n_negative, n_zeros)

    return (
        _get_elem_at_rank(middle - 1, data, n_negative, n_zeros)
        + _get_elem_at_rank(middle, data, n_negative, n_zeros)
    ) / 2.0


def csc_median_axis_0(X):
    """Find the median across axis 0 of a CSC matrix.
    It is equivalent to doing np.median(X, axis=0).
    Parameters
    ----------
    X : CSC sparse matrix, shape (n_samples, n_features)
        Input data.
    Returns
    -------
    median : ndarray, shape (n_features,)
        Median.
    """
    if not isinstance(X, sparse.csc_matrix):
        raise TypeError("Expected matrix of CSC format, got %s" % X.format)

    indptr = X.indptr
    n_samples, n_features = X.shape
    median = np.zeros(n_features)

    for f_ind, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):

        # Prevent modifying X in place
        data = np.copy(X.data[start:end])
        nz = n_samples - data.size
        median[f_ind] = _get_median(data, nz)

    return median

def sparse_var_axis0(X):
    """
    Compute variance across axis=0 (per gene) for sparse CSR/CSC matrix.
    Equivalent to np.var(X.toarray(), axis=0) but memory-efficient.
    """
    if not sparse.issparse(X):
        return np.var(X, axis=0)

    # Ensure CSR format for efficient row slicing
    X = sparse.csr_matrix(X)

    n = X.shape[0]  # number of cells
    if n <= 1:
        return np.zeros(X.shape[1])

    # Mean = (sum of values) / n
    sum_x = np.array(X.sum(axis=0)).ravel()  # shape: (n_genes,)
    mean_x = sum_x / n

    # E[X^2] = sum(x_ij^2) / n
    sum_x2 = np.array((X.multiply(X)).sum(axis=0)).ravel()
    mean_x2 = sum_x2 / n

    # Var(X) = E[X^2] - (E[X])^2
    var_x = mean_x2 - mean_x ** 2
    return var_x
