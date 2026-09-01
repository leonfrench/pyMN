import numpy as np
import bottleneck
from scipy import sparse
import pandas as pd

from .utils import format_bytes, numerical_thread_summary


def _select_genes_from_statistics(median, variance):
    """Select genes from per-gene median and variance vectors."""
    # ``interpolation`` retains compatibility with the NumPy 1.21 environment
    # used by the baseline as well as newer NumPy releases.
    bins = np.quantile(median,
                       q=np.linspace(0, 1, 11),
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


def compute_var_genes_batched(X, row_indices, gene_batch_size=1024):
    """Compute variable genes while bounding temporary data by gene batches.

    Unlike ``compute_var_genes``, this helper never constructs an AnnData view
    containing every gene for a study. Sparse row subsets are materialized only
    for the current gene batch.
    """
    if gene_batch_size <= 0:
        raise ValueError("gene_batch_size must be a positive integer")

    row_indices = np.asarray(row_indices, dtype=np.intp)
    n_genes = X.shape[1]
    median = np.empty(n_genes, dtype=float)
    variance = np.empty(n_genes, dtype=float)

    for start in range(0, n_genes, gene_batch_size):
        stop = min(start + gene_batch_size, n_genes)
        if sparse.issparse(X):
            block = X[row_indices, start:stop]
            median[start:stop] = csc_median_axis_0(
                sparse.csc_matrix(block)
            )
            variance[start:stop] = sparse_var_axis0(block)
        else:
            block = np.asarray(X[np.ix_(row_indices, np.arange(start, stop))])
            median[start:stop] = bottleneck.median(block, axis=0)
            variance[start:stop] = np.var(block, axis=0)

    return _select_genes_from_statistics(median, variance)


def variableGenes(adata,
                  study_col,
                  return_vect=False,
                  memory_constrained=False,
                  gene_batch_size=1024):
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

    Returns:
        np.ndarray -- None if saving in adata.var['highly_variable'], array of booleans if returning of length ngenes
    """

    assert study_col in adata.obs.columns, f"Study col '{study_col}' not in adata.obs"


    studies = np.unique(adata.obs[study_col])
    if memory_constrained:
        if gene_batch_size <= 0:
            raise ValueError("gene_batch_size must be a positive integer")
        var_genes = np.ones(adata.n_vars, dtype=bool)
        study_values = adata.obs[study_col].values
        study_sizes = np.asarray([
            np.count_nonzero(study_values == study) for study in studies
        ])
        input_itemsize = np.dtype(adata.X.dtype).itemsize
        batch_genes = min(gene_batch_size, adata.n_vars)
        dense_value_batch = (
            int(study_sizes.max()) * batch_genes * input_itemsize
        )
        print(
            "[variableGenes memory_constrained] setup\n"
            f"  cells={adata.n_obs:,}, genes={adata.n_vars:,}, "
            f"studies={len(studies):,}\n"
            f"  gene_batch_size={gene_batch_size:,}, "
            f"largest_study_cells={int(study_sizes.max()):,}\n"
            "  estimated_dense_value_batch="
            f"{format_bytes(dense_value_batch)} "
            "(sparse temporary storage depends on nnz)\n"
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
