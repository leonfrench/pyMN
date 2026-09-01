import numpy as np
import pandas as pd
import bottleneck
import warnings
import os
import shutil
import tempfile
import time
from scipy import sparse

import gc

from .utils import *



def MetaNeighborUS(adata,
                   study_col,
                   ct_col,
                   var_genes="highly_variable",
                   symmetric_output=True,
                   node_degree_normalization=True,
                   fast_version=False,
                   one_vs_best=False,
                   trained_model=None,
                   save_uns=True,
                   compute_p=False,
                   mn_key="MetaNeighborUS",
                   memory_constrained=False,
                   cell_batch_size=256,
                   score_batch_size=64,
                   temporary_directory=None,
                   vote_dtype="float64"):
    """Runs Unsupervised version of MetaNeighbor

    When it is difficult to know how cell type labels compare across datasets this
    function helps users to make an educated guess about the overlaps without
    requiring in-depth knowledge of marker genes.


    The output is a cell type-by-cell type mean AUROC matrix, which is
    built by treating each pair of cell types as testing and training data for
    MetaNeighbor, then taking the average AUROC for each pair (NB scores will not
    be identical because each test cell type is scored out of its own dataset,
    and the differential heterogeneity of datasets will influence scores).
    If symmetric_output is set to FALSE, the training cell types are displayed
    as columns and the test cell types are displayed as rows.
    If trained_model was provided, the output will be a cell type-by-cell
    type AUROC matrix with training cell types as columns and test cell types
    as rows (no swapping of test and train, no averaging).

    Arguments:
       adata {AnnData} -- AnnData object containing all the single cell experiements concatenated together
       study_col {str} -- String referencing column in andata.obs that identifies study label for datasets
       ct_col {str} -- String referencing column in andata.obs that identifies cellt type labels

    Keyword Arguments:
        var_genes {str or vector} -- String for boolean column in adata.var that indicates highly variable
            genes or vector of highly variable genes (default: {'highly_variable'})
        symmetric_output {bool} --  Boolean indicating whether make square matrix output symmetric (default: {True})
        node_degree_normalization {bool} -- Boolean indicating whether to normalize votes by node degree (default: {True})
        fast_version {bool} -- boolean indicating whether to run fast approximate version (default: {False})
        one_vs_best {bool} --  boolean indicating whether to compute AUROCs as one vs best instead
            of one vs all (must also have fast_version = True or use pretrained model) (default: {False})
        trained_model {pd.DataFrame} -- A dataframe containing a trained model from pymn.trainModel
            or from the R vesion of MetaNeighbor::trainModel (default: {None})
        save_uns {bool} -- Boolean indicating whether to save in adata.uns[mn_key],
            when False returns cell type x cell type AUROCs dataframe (default: {True})
        mn_key {str} -- Key for saving in adata.uns[mn_key] (default: {'MetaNeighborUS'})
        memory_constrained {bool} -- Use the batched fast implementation and a
            temporary memory map for votes (default: {False})
        cell_batch_size {int} -- Maximum cells normalized together when
            memory_constrained is True (default: {256})
        score_batch_size {int} -- Maximum vote columns ranked together when
            memory_constrained is True (default: {64})
        temporary_directory {str or PathLike} -- Parent directory for temporary
            memory-mapped vote files (default: {None})
        vote_dtype {str or numpy dtype} -- Floating dtype for normalized values,
            centroids, and temporary votes; float64 matches the legacy path most
            closely and float32 further reduces memory and disk use (default: {'float64'})
    """

    assert study_col in adata.obs_keys(), "Study Col not in adata"
    assert ct_col in adata.obs_keys(), "Cluster Col not in adata"

    if trained_model is not None:
        var_genes = adata.var_names[np.in1d(adata.var_names,
                                            trained_model.index)]
        trained_model = pd.concat([
            pd.DataFrame(trained_model.iloc[0]).T, trained_model.loc[var_genes]
        ])
    elif type(var_genes) is str:
        assert (
            var_genes in adata.var_keys()
        ), f"If passing a string ({var_genes}) for var names, it must be in adata.var_keys()"
        var_genes = adata.var_names[adata.var[var_genes]]
    else:
        var_genes = adata.var_names[np.in1d(adata.var_names, var_genes)]

    assert var_genes.shape[0] > 2, "Must have at least 2 genes"
    if var_genes.shape[0] < 5:
        warnings.warn("You should have at least 5 Variable Genes",
                      category=UserWarning)
    if one_vs_best:
        assert (fast_version or trained_model is not None
                ), "If you want to run in one_vs_best mode you must also \
         run in fast version mode or use a pretrained model"
    if memory_constrained:
        if trained_model is not None:
            raise NotImplementedError(
                "memory_constrained currently supports de novo fast_version runs only"
            )
        if not fast_version:
            raise ValueError(
                "memory_constrained=True currently requires fast_version=True"
            )

    if trained_model is None:
        assert (np.unique(adata.obs[study_col].values).shape[0] >
                1), "Need more than 1 study"
        if fast_version:
            # Fast verion doesn't work with Categorical datatype
            if memory_constrained:
                gene_indices = adata.var_names.get_indexer(var_genes)
                cell_nv = metaNeighborUS_fast_memory_constrained(
                    adata.X,
                    adata.obs[study_col],
                    adata.obs[ct_col],
                    gene_indices,
                    node_degree_normalization,
                    one_vs_best,
                    compute_p,
                    cell_batch_size=cell_batch_size,
                    score_batch_size=score_batch_size,
                    temporary_directory=temporary_directory,
                    vote_dtype=vote_dtype,
                )
            else:
                assert (
                    adata.obs[study_col].dtype.name != "category"
                ), "Study Col is a category type, cast to either string or int"
                assert (
                    adata.obs[ct_col].dtype.name != "category"
                ), "Cell Type Col is a category type, cast to either string or int"

                cell_nv = metaNeighborUS_fast(adata[:, var_genes].X,
                                              adata.obs[study_col],
                                              adata.obs[ct_col],
                                              node_degree_normalization,
                                              one_vs_best, compute_p)
        else:
            cell_nv = metaNeighborUS_default(adata[:, var_genes], study_col,
                                             ct_col, node_degree_normalization,
                                             compute_p)
    else:

        cell_nv = MetaNeighborUS_from_trained(trained_model,
                                              adata[:, var_genes].X,
                                              adata.obs[study_col].values,
                                              adata.obs[ct_col].values,
                                              node_degree_normalization,
                                              one_vs_best, compute_p)
    if compute_p:
        cell_p = cell_nv[1]
        cell_nv = cell_nv[0]
        cell_p = cell_p.astype(float)

    cell_nv = cell_nv.astype(float)
    if symmetric_output and not one_vs_best:
        cell_nv = (cell_nv + cell_nv.T) / 2
    if save_uns:
        if one_vs_best:
            adata.uns[f"{mn_key}_1v1"] = cell_nv
        else:
            adata.uns[mn_key] = cell_nv
        adata.uns[f"{mn_key}_params"] = {
            "fast": fast_version,
            "node_degree_normalization": node_degree_normalization,
            "study_col": study_col,
            "ct_col": ct_col,
            "one_vs_best": one_vs_best,
            "symmetric_output": symmetric_output,
            "memory_constrained": memory_constrained,
            "cell_batch_size": cell_batch_size,
            "score_batch_size": score_batch_size,
            "temporary_directory": None if temporary_directory is None else str(temporary_directory),
            "vote_dtype": str(np.dtype(vote_dtype)),
        }
        if compute_p:
            adata.uns[f'{mn_key}_pval'] = cell_p
    else:
        return cell_nv


def metaNeighborUS_default(adata, study_col, ct_col, node_degree_normalization,
                           compute_p):
    """Runs MetaNeighbor using Default Method



    Arguments:
        adata {AnnData} -- AnnData object containing all the single cell experiements concatenated together
        study_col {str} -- String referencing column in andata.obs that identifies study label for datasets
        ct_col {str} -- String referencing column in andata.obs that identifies cellt type labels
        node_degree_normalization {bool} -- Boolean indicating whether to normalize votes by node degree

    Returns:
        pd.DataFrame -- ROCs for cell type x cell type labels
    """
    pheno, cell_labels, study_ct_uniq = create_cell_labels(
        adata, study_col, ct_col)

    rank_data = create_nw_spearman(adata.X.T)

    sum_in = rank_data @ cell_labels.values

    if node_degree_normalization:
        sum_all = np.sum(rank_data, axis=0)
        sum_in /= sum_all[:, None]

    cell_nv = compute_aurocs_default(sum_in, study_ct_uniq, pheno, study_col,
                                     ct_col, compute_p)
    return cell_nv


def compute_aurocs_default(sum_in, study_ct_uniq, pheno, study_col, ct_col,
                           compute_p):
    """Helper function to compute AUROCs from votes matrix of cells


    Arguments:
        sum_in {np.ndarray} -- votes matrix, cells x cell types votes
        study_ct_uniq {vector} -- vector of study_id|cell_type labels
        pheno {pd.DataFrame} -- dataframe wtih study_ct, study_id and ct_col for all cells
        study_col {str} -- String name of study_col in pheno
        ct_col {str} -- Stirng name of cell type col in pheno

    Returns:
        pd.DataFrame -- ROCs for cell type x cell type labels
    """
    cell_nv = pd.DataFrame(index=study_ct_uniq)
    if compute_p:
        cell_p = pd.DataFrame(index=study_ct_uniq)
    for ct in study_ct_uniq:
        predicts_tmp = sum_in.copy()
        study, cellT = (pheno[pheno.study_ct == ct].drop_duplicates()[[
            study_col, ct_col
        ]].values[0])  # Don't want to split string in case of charcter issues
        slicer = pheno[study_col] == study
        pheno2 = pheno[slicer]
        predicts_tmp = predicts_tmp[slicer]
        predicts_tmp = bottleneck.nanrankdata(predicts_tmp, axis=0)

        filter_mat = np.zeros_like(predicts_tmp)
        filter_mat[pheno2.study_ct == ct] = 1

        predicts_tmp[filter_mat == 0] = 0

        n_p = bottleneck.nansum(filter_mat, axis=0)
        nn = filter_mat.shape[0] - n_p
        p = bottleneck.nansum(predicts_tmp, axis=0)
        roc = (p / n_p - (n_p + 1) / 2) / nn
        cell_nv[ct] = roc
        if compute_p:
            U = roc * n_p * nn
            Z = (np.abs(U - (n_p * nn / 2))) / np.sqrt(n_p * nn *
                                                       (n_p + nn + 1) / 12)
            P = stats.norm.sf(Z)
            cell_p[ct] = P
        del predicts_tmp, filter_mat
        gc.collect()
    if compute_p:
        return cell_nv, cell_p
    return cell_nv



def metaNeighborUS_fast(X, S, C, node_degree_normalization, one_vs_best,
                        compute_p):

    """Fast MetaNeighbor Approximation Helper function


    The fast version is vectorized according to the following equations
    (Note that the point of these equations is to *never* compute the cell-cell network
     by reordering the matrix operations):
     - INPUTS:
       + Q = test (Query) data (genes x cells)
       + R = train (Ref) data (genes x cells)
       + L = binary encoding of training cell types (Labels) (cells x cell types)
       + S = binary encoding of train Studies (cells x studies)
     - NOTATIONS:
       + X* = normalize_cols(X) ~ scale(colRanks(X)) denotes normalized data
              (Spearman correlation becomes a simple dot product on normalized data)
       + N = Spearman(Q,R) = t(Q*).R* is the cell-cell similarity network
       + CL = R*.L are the cell type centroids (in the normalized space)
       + CS = R*.S are the study centroids (in the normalized space)
       + 1.L = colSums(L) = number of cells per (train) cell type
       + 1.S = colSums(S) = number of cells per (train) study
     - WITHOUT node degree normalization
       + Votes = N.L = t(Q*).R*.L = t(Q*).CL
     - WITH node degree normalization
       + Network becomes N+1 to avoid negative values
       + Votes = (N+1).L = N.L + 1.L = t(Q*).CL + 1.L
       + Node degree = (N+1).S = t(Q*).CS + 1.S
       + Note: Node degree is computed independently for each train study.

    Arguments:
        X {array} -- cells x variableGenes matrix (dense or sparse)
        S {vector} -- vector of study_id labels
        C {vector} -- vector of cell type labels
        node_degree_normalization {bool} -- Boolean indicating whether to normalize votes by node degree
        one_vs_best {bool} -- Boolean indicating whether to compute one vs best if True, one vs all if False

    Returns:
        pd.DataFrame --  ROCs for cell type x cell type labels
    """

    # Makes it genes X cells
    X_norm = np.asfortranarray(normalize_cells(X).T)

    # Remove cells that have no variance
    filter_cells = np.any(np.isnan(X_norm), axis=0)
    X_norm = X_norm[:, ~filter_cells]

    S = S[~filter_cells]
    C = C[~filter_cells]
    S_order = np.unique(S.values)
    C_order = np.unique(C.values)

    labels = join_labels(S.values, C.values)
    labels_matrix = design_matrix(labels)

    cluster_centroids = X_norm @ labels_matrix.values
    cluster_centroids = pd.DataFrame(cluster_centroids,
                                     columns=labels_matrix.columns)
    labels_order = labels_matrix.columns
    n_cells_per_cluster = np.sum(labels_matrix.values, axis=0)
    LSC = pd.DataFrame({"study": S.values, "cluster": C.values}, index=labels)


    result = predict_and_score(X_norm, LSC, cluster_centroids,
                               n_cells_per_cluster, labels_order,
                               node_degree_normalization, one_vs_best,
                               compute_p)
    if compute_p:
        aurocs = result[0]
        aurocs = aurocs[aurocs.index]

        p_vals = result[1]
        p_vals = p_vals[p_vals.index]
        return aurocs, p_vals


    result = result[result.index]
    return result


def _validate_memory_constrained_parameters(cell_batch_size,
                                            score_batch_size,
                                            vote_dtype,
                                            temporary_directory):
    if cell_batch_size <= 0:
        raise ValueError("cell_batch_size must be a positive integer")
    if score_batch_size <= 0:
        raise ValueError("score_batch_size must be a positive integer")
    dtype = np.dtype(vote_dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("vote_dtype must be float32 or float64")
    if temporary_directory is not None:
        temporary_directory = os.fspath(temporary_directory)
        if not os.path.isdir(temporary_directory):
            raise FileNotFoundError(
                f"temporary_directory does not exist: {temporary_directory}"
            )
    return dtype, temporary_directory


def _encode_study_cell_types(studies, cell_types):
    """Encode study/cell-type pairs without one joined string per cell."""
    studies = np.asarray(studies)
    cell_types = np.asarray(cell_types)
    study_levels, study_codes = np.unique(studies, return_inverse=True)
    cell_type_levels, cell_type_codes = np.unique(cell_types,
                                                   return_inverse=True)

    pair_keys = study_codes.astype(np.int64, copy=False)
    pair_keys = pair_keys * len(cell_type_levels) + cell_type_codes
    observed_keys, pair_codes = np.unique(pair_keys, return_inverse=True)
    pair_study_codes = observed_keys // len(cell_type_levels)
    pair_cell_type_codes = observed_keys % len(cell_type_levels)
    pair_labels = np.asarray([
        f"{study_levels[study]}|{cell_type_levels[cell_type]}"
        for study, cell_type in zip(pair_study_codes, pair_cell_type_codes)
    ])

    # The legacy final result is ordered by sorted study and then by sorted
    # cell type within that study (predict_and_score concatenates by study and
    # finally reorders columns to match those rows).
    return pair_codes, pair_labels, study_levels[pair_study_codes]


def _get_expression_block(X, row_selector, gene_indices):
    if sparse.issparse(X):
        return X[row_selector, :][:, gene_indices].toarray()

    if isinstance(row_selector, slice):
        rows = np.arange(*row_selector.indices(X.shape[0]), dtype=np.intp)
    else:
        rows = np.asarray(row_selector, dtype=np.intp)
    return np.asarray(X[np.ix_(rows, gene_indices)])


def _normalize_cell_batch(X, row_selector, gene_indices, dtype):
    """Rank-normalize a bounded batch, matching normalize_cells semantics."""
    dense = _get_expression_block(X, row_selector, gene_indices)
    normalized = bottleneck.rankdata(dense, axis=1).astype(dtype, copy=False)
    average = np.mean(normalized, axis=1)
    normalized -= average[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.sqrt(bottleneck.nansum(normalized**2, axis=1))[:, None]
        normalized /= norm
    return normalized


def _build_cluster_centroids_batched(X,
                                     gene_indices,
                                     pair_codes,
                                     n_pairs,
                                     cell_batch_size,
                                     dtype):
    n_cells = X.shape[0]
    n_genes = len(gene_indices)
    cluster_centroids = np.zeros((n_pairs, n_genes), dtype=dtype)
    valid_cells = np.zeros(n_cells, dtype=bool)

    for start in range(0, n_cells, cell_batch_size):
        stop = min(start + cell_batch_size, n_cells)
        normalized = _normalize_cell_batch(
            X, slice(start, stop), gene_indices, dtype
        )
        valid = ~np.any(np.isnan(normalized), axis=1)
        valid_cells[start:stop] = valid
        if np.any(valid):
            values = normalized if np.all(valid) else normalized[valid]
            batch_codes = pair_codes[start:stop][valid]
            present_codes, local_codes = np.unique(
                batch_codes, return_inverse=True
            )
            membership = sparse.csr_matrix(
                (
                    np.ones(len(local_codes), dtype=dtype),
                    (np.arange(len(local_codes)), local_codes),
                ),
                shape=(len(local_codes), len(present_codes)),
            )
            cluster_centroids[present_codes] += membership.T @ values

    n_cells_per_cluster = np.bincount(
        pair_codes[valid_cells], minlength=n_pairs
    ).astype(dtype, copy=False)
    return cluster_centroids, n_cells_per_cluster, valid_cells


def _compute_batched_aurocs(vote_block,
                            positive_matrix,
                            n_positive,
                            compute_p):
    ranks = bottleneck.rankdata(vote_block, axis=0)
    sum_positive_ranks = np.asarray(positive_matrix.T @ ranks)
    n_negative = positive_matrix.shape[0] - n_positive
    roc = sum_positive_ranks / n_positive[:, None]
    roc -= (n_positive[:, None] + 1) / 2
    roc /= n_negative[:, None]

    if not compute_p:
        return roc
    n_positive_2d = n_positive[:, None]
    n_negative_2d = n_negative[:, None]
    U = roc * n_positive_2d * n_negative_2d
    Z = np.abs(U - (n_positive_2d * n_negative_2d / 2))
    Z /= np.sqrt(
        n_positive_2d
        * n_negative_2d
        * (n_positive_2d + n_negative_2d + 1)
        / 12
    )
    return roc, stats.norm.sf(Z)


def _fill_vote_memmap(votes,
                      X,
                      test_cell_indices,
                      gene_indices,
                      cluster_centroids,
                      n_cells_per_cluster,
                      cluster_study_codes,
                      study_centroids,
                      n_cells_per_study,
                      node_degree_normalization,
                      cell_batch_size,
                      dtype):
    for start in range(0, len(test_cell_indices), cell_batch_size):
        stop = min(start + cell_batch_size, len(test_cell_indices))
        row_indices = test_cell_indices[start:stop]
        normalized = _normalize_cell_batch(
            X, row_indices, gene_indices, dtype
        )
        batch_votes = normalized @ cluster_centroids.T
        if node_degree_normalization:
            batch_votes += n_cells_per_cluster
            node_degree = normalized @ study_centroids.T
            node_degree += n_cells_per_study
            for train_study_code in range(study_centroids.shape[0]):
                is_train = cluster_study_codes == train_study_code
                batch_votes[:, is_train] /= node_degree[:,
                                                        train_study_code][:,
                                                                          None]
        votes[start:stop, :] = batch_votes
    votes.flush()


def metaNeighborUS_fast_memory_constrained(
        X,
        S,
        C,
        gene_indices,
        node_degree_normalization,
        one_vs_best,
        compute_p,
        cell_batch_size=256,
        score_batch_size=64,
        temporary_directory=None,
        vote_dtype="float64"):
    """Fast MetaNeighborUS with bounded in-memory cell and score batches."""
    dtype, temporary_directory = _validate_memory_constrained_parameters(
        cell_batch_size,
        score_batch_size,
        vote_dtype,
        temporary_directory,
    )
    gene_indices = np.asarray(gene_indices, dtype=np.intp)
    study_values = np.asarray(S)
    pair_codes, pair_labels, pair_studies = _encode_study_cell_types(S, C)
    started_at = time.perf_counter()
    temporary_parent = os.path.abspath(
        tempfile.gettempdir()
        if temporary_directory is None
        else temporary_directory
    )
    _, study_cell_counts = np.unique(study_values, return_counts=True)
    largest_study_cells = int(study_cell_counts.max())
    n_cells = X.shape[0]
    n_genes = len(gene_indices)
    n_pairs = len(pair_labels)
    batch_cells = min(cell_batch_size, n_cells)
    input_itemsize = np.dtype(X.dtype).itemsize
    normalization_batch_bytes = batch_cells * n_genes * max(
        input_itemsize + np.dtype("float64").itemsize,
        input_itemsize + 2 * dtype.itemsize,
    )
    centroid_bytes = n_pairs * n_genes * dtype.itemsize
    vote_file_upper_bound = largest_study_cells * n_pairs * dtype.itemsize
    score_columns = min(score_batch_size, n_pairs)
    score_batch_bytes = (
        largest_study_cells
        * score_columns
        * (dtype.itemsize + np.dtype("float64").itemsize)
    )
    result_multiplier = 2 if compute_p else 1
    result_bytes = n_pairs * n_pairs * np.dtype("float64").itemsize
    result_bytes *= result_multiplier
    mode = "one_vs_best" if one_vs_best else "all_vs_all"
    print(
        "[MetaNeighborUS memory_constrained] setup\n"
        f"  mode={mode}, cells={n_cells:,}, selected_genes={n_genes:,}, "
        f"studies={len(np.unique(study_values)):,}, "
        f"study_cell_types={n_pairs:,}\n"
        f"  cell_batch_size={cell_batch_size:,}, "
        f"score_batch_size={score_batch_size:,}, vote_dtype={dtype.name}\n"
        f"  temporary_directory={temporary_parent}\n"
        f"  estimated_max_vote_file={format_bytes(vote_file_upper_bound)}, "
        f"centroids={format_bytes(centroid_bytes)}, "
        f"results={format_bytes(result_bytes)}\n"
        "  estimated_peak_work_batches: "
        f"normalization={format_bytes(normalization_batch_bytes)}, "
        f"score_ranking={format_bytes(score_batch_bytes)}\n"
        f"  cores: {numerical_thread_summary()}",
        flush=True,
    )
    print(
        "[MetaNeighborUS memory_constrained] building cluster centroids",
        flush=True,
    )

    cluster_centroids, n_cells_per_cluster, valid_cells = (
        _build_cluster_centroids_batched(
            X,
            gene_indices,
            pair_codes,
            len(pair_labels),
            cell_batch_size,
            dtype,
        )
    )
    print(
        "[MetaNeighborUS memory_constrained] centroids complete: "
        f"valid_cells={int(valid_cells.sum()):,}/{n_cells:,}",
        flush=True,
    )

    # Match the legacy behavior of removing cells with undefined variance,
    # including removal of any study/cell-type labels left without cells.
    keep_pairs = n_cells_per_cluster > 0
    if not np.all(keep_pairs):
        old_to_new = np.full(len(keep_pairs), -1, dtype=np.intp)
        old_to_new[keep_pairs] = np.arange(np.count_nonzero(keep_pairs))
        pair_codes = old_to_new[pair_codes]
        pair_labels = pair_labels[keep_pairs]
        pair_studies = pair_studies[keep_pairs]
        cluster_centroids = cluster_centroids[keep_pairs]
        n_cells_per_cluster = n_cells_per_cluster[keep_pairs]

    study_order = np.unique(pair_studies)
    cluster_study_codes = np.searchsorted(study_order, pair_studies)
    study_centroids = np.zeros(
        (len(study_order), cluster_centroids.shape[1]), dtype=dtype
    )
    np.add.at(study_centroids, cluster_study_codes, cluster_centroids)
    n_cells_per_study = np.bincount(
        cluster_study_codes,
        weights=n_cells_per_cluster,
        minlength=len(study_order),
    ).astype(dtype, copy=False)

    n_pairs = len(pair_labels)
    result = np.full((n_pairs, n_pairs), np.nan, dtype=float)
    result_p = (
        np.full((n_pairs, n_pairs), np.nan, dtype=float)
        if compute_p
        else None
    )

    for study_number, test_study in enumerate(study_order, start=1):
        test_cell_indices = np.flatnonzero(valid_cells
                                           & (study_values == test_study))
        test_pair_indices = np.flatnonzero(pair_studies == test_study)
        global_to_local = np.full(n_pairs, -1, dtype=np.intp)
        global_to_local[test_pair_indices] = np.arange(len(test_pair_indices))
        local_cell_codes = global_to_local[pair_codes[test_cell_indices]]
        n_positive = np.bincount(
            local_cell_codes, minlength=len(test_pair_indices)
        ).astype(float, copy=False)
        positive_matrix = sparse.csr_matrix(
            (
                np.ones(len(test_cell_indices), dtype=float),
                (np.arange(len(test_cell_indices)), local_cell_codes),
            ),
            shape=(len(test_cell_indices), len(test_pair_indices)),
        )

        required_vote_bytes = (
            len(test_cell_indices) * n_pairs * dtype.itemsize
        )
        available_vote_bytes = shutil.disk_usage(temporary_parent).free
        if required_vote_bytes > available_vote_bytes:
            raise OSError(
                "Insufficient temporary storage for memory-constrained votes: "
                f"need {required_vote_bytes} bytes for study {test_study!r}, "
                f"but {available_vote_bytes} bytes are available in "
                f"{temporary_parent!r}"
            )
        print(
            "[MetaNeighborUS memory_constrained] "
            f"study {study_number}/{len(study_order)}: {test_study!s}, "
            f"cells={len(test_cell_indices):,}, "
            f"vote_file={format_bytes(required_vote_bytes)}, "
            f"free_scratch={format_bytes(available_vote_bytes)}",
            flush=True,
        )

        with tempfile.TemporaryDirectory(
                prefix="pymn-votes-", dir=temporary_directory) as work_dir:
            vote_path = os.path.join(work_dir, "votes.dat")
            votes = np.memmap(
                vote_path,
                mode="w+",
                dtype=dtype,
                shape=(len(test_cell_indices), n_pairs),
                order="F",
            )
            _fill_vote_memmap(
                votes,
                X,
                test_cell_indices,
                gene_indices,
                cluster_centroids,
                n_cells_per_cluster,
                cluster_study_codes,
                study_centroids,
                n_cells_per_study,
                node_degree_normalization,
                cell_batch_size,
                dtype,
            )

            categorical_index = None
            if one_vs_best:
                categorical_index = pd.CategoricalIndex(
                    pd.Categorical.from_codes(
                        local_cell_codes,
                        categories=pair_labels[test_pair_indices],
                    )
                )

            for score_start in range(0, n_pairs, score_batch_size):
                score_stop = min(score_start + score_batch_size, n_pairs)
                score_columns = np.arange(score_start, score_stop)
                vote_block = np.array(
                    votes[:, score_start:score_stop], copy=True, order="F"
                )
                block_result = _compute_batched_aurocs(
                    vote_block,
                    positive_matrix,
                    n_positive,
                    compute_p,
                )
                if compute_p:
                    block_aurocs, block_p = block_result
                    result_p[np.ix_(test_pair_indices,
                                    score_columns)] = block_p
                else:
                    block_aurocs = block_result

                if one_vs_best:
                    votes_dataframe = pd.DataFrame(
                        vote_block,
                        index=categorical_index,
                        columns=pair_labels[score_columns],
                    )
                    aurocs_dataframe = pd.DataFrame(
                        block_aurocs,
                        index=pair_labels[test_pair_indices],
                        columns=pair_labels[score_columns],
                    )
                    block_aurocs = compute_1v1_aurocs(
                        votes_dataframe, aurocs_dataframe
                    ).to_numpy(dtype=float)

                result[np.ix_(test_pair_indices,
                              score_columns)] = block_aurocs

            del votes
            gc.collect()
        print(
            "[MetaNeighborUS memory_constrained] "
            f"study {study_number}/{len(study_order)} complete; "
            "temporary vote file removed",
            flush=True,
        )

    result = pd.DataFrame(result, index=pair_labels, columns=pair_labels)
    elapsed = time.perf_counter() - started_at
    print(
        "[MetaNeighborUS memory_constrained] complete: "
        f"mode={mode}, elapsed={elapsed:.2f}s, studies={len(study_order):,}",
        flush=True,
    )
    if compute_p:
        result_p = pd.DataFrame(result_p,
                                index=pair_labels,
                                columns=pair_labels)
        return result, result_p
    return result



def predict_and_score(X_test,
                      LSC,
                      cluster_centroids,
                      n_cells_per_cluster,
                      labels_order,
                      node_degree_normalization,
                      one_vs_best,
                      compute_p,
                      pretrained=False):

    """[summary]

    [description]

    Arguments:
        X_test {np.ndarray} -- Normalized gene x cell expression
        LSC {pd.DataFrame} -- Dataframe with columns of study_col and ct_col and study_col|ct_col as index
        cluster_centroids {pd.DataFrame} -- Dataframe with genes x cell type centroids
        n_cells_per_cluster {vector} -- Vector in same order as cluster_centroids columns for number of cells per cluster
        labels_order {vector} -- Vector order for study_col|ct_col labels
        node_degree_normalization {bool} -- Boolean indicating whether to normalize votes by node degree
        one_vs_best {bool} -- Boolean indicating whether to compute one vs best if True, one vs all if False

    Keyword Arguments:
        pretrained {bool} -- Whether or not it is passing a pretrained model or not (default: {False})

    Returns:
        pd.DataFrame -- ROCs for cell type x cell type labels
    """
    if node_degree_normalization:
        if pretrained:
            get_study_id = np.vectorize(lambda x: x.split("|")[0])
            centroid_study_label = get_study_id(
                cluster_centroids.columns.values)
        else:
            centroid_study_label = (LSC.drop_duplicates().loc[labels_order,
                                                              "study"].values)
        study_matrix = design_matrix(centroid_study_label)
        train_study_id = study_matrix.columns
        study_centroids = cluster_centroids.values @ study_matrix.values
        n_cells_per_study = n_cells_per_cluster @ study_matrix.values

    result = []
    if compute_p:
        result_p = []
    S = LSC["study"].values
    for test_study in np.unique(S):
        is_test = S == test_study
        X_dataset = X_test[:, is_test]
        votes_idx = LSC.index[is_test]
        votes_cols = labels_order
        votes = np.asfortranarray(X_dataset.T @ cluster_centroids.values)
        if node_degree_normalization:
            votes += n_cells_per_cluster

            node_degree = np.asfortranarray(X_dataset.T @ study_centroids)
            node_degree += n_cells_per_study

            for train_study in np.unique(train_study_id):
                is_train = centroid_study_label == train_study
                norm = node_degree[:, train_study_id == train_study]
                votes[:, is_train] = votes[:, is_train] / norm
        votes = pd.DataFrame(votes, index=votes_idx, columns=votes_cols)


        aurocs = compute_aurocs(votes,
                                positives=design_matrix(votes.index),
                                compute_p=compute_p)
        if compute_p:
            result_p.append(aurocs[1])
            aurocs = aurocs[0]

        if one_vs_best:
            aurocs = compute_1v1_aurocs(votes, aurocs)
        result.append(aurocs)
    if compute_p:
        return pd.concat(result), pd.concat(result_p)
    return pd.concat(result)



def MetaNeighborUS_from_trained(trained_model, test_data, study_col, ct_col,
                                node_degree_normalization, one_vs_best,
                                compute_p):

    """MetaNeighbor from Pretrained model

    Runs MetaNeighbor using a pretrained model in the fast approximate version

    Arguments:
        trained_model {pd.DataFrame} -- Genes x Cell Type dataframe of model, with first row being number of cells per cell type in the model
        test_data {array} -- Genes x Cells expression data
        study_col {vector} -- vector of study_id labels
        ct_col {vector} -- vector of cell type labels
        node_degree_normalization {bool} -- Boolean indicating whether to normalize votes by node degree
        one_vs_best {bool} -- Boolean indicating whether to compute one vs best if True, one vs all if False

    Returns:
        pd.DataFrame -- ROCs for cell type x cell type labels
    """
    dat = normalize_cells(test_data).T
    is_na = np.any(np.isnan(dat), axis=0)
    dat = dat[:, ~is_na]
    cluster_centroids = trained_model.iloc[1:]
    n_cells_per_cluster = trained_model.iloc[0].values
    study_col = study_col[~is_na]
    ct_col = ct_col[~is_na]
    labels = join_labels(study_col, ct_col, replace_bar=True)
    LSC = pd.DataFrame({"study": study_col, "cluster": ct_col}, index=labels)
    result = predict_and_score(
        dat,
        LSC,
        cluster_centroids,
        n_cells_per_cluster,
        cluster_centroids.columns,
        node_degree_normalization,
        one_vs_best,
        compute_p,
        pretrained=True,
    )
    return result
