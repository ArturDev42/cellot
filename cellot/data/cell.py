#!/usr/bin/python3

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from cellot.models import load_autoencoder_model
from cellot.utils import load_config
from cellot.data.utils import cast_dataset_to_loader
from cellot.utils.helpers import nest_dict


class AnnDataDataset(Dataset):
    def __init__(
        self, adata, obs=None, categories=None, include_index=False, dim_red=None
    ):
        self.adata = adata
        self.adata.X = self.adata.X.astype(np.float32)
        self.obs = obs
        self.categories = categories
        self.include_index = include_index

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        value = self.adata.X[idx]

        if self.obs is not None:
            meta = self.categories.index(self.adata.obs[self.obs].iloc[idx])
            value = value, int(meta)

        if self.include_index:
            return self.adata.obs_names[idx], value

        return value


def read_list(arg):

    if isinstance(arg, str):
        arg = Path(arg)
        assert arg.exists()
        lst = arg.read_text().split()
    else:
        lst = arg

    return list(lst)


def read_single_anndata(config: str) -> anndata.AnnData:
    """## Reads anndata file that is used in load_cell_data() which creates a dataloader that is
    used to train the model. We use this function instead of the function from the cellot authors,
    because we do our own data pre-processing and provide the already pre-processed file to cellot.
    If no other functions in the cellot codebase are changed, cellot expects the anndata file to
    have a column called 'split' containing train/test/val and a column called 'transport' containing
    the label 'source' for control and the label 'target' for perturbations.

    ### Args:
        - `config (yaml)`: Contains the path to the anndata file.
    """
    path = data.config.path
    data = anndata.read(path)

    return data


def load_cell_data(
    config,
    data=None,
    split_on=None,
    return_as="loader",
    include_model_kwargs=False,
    pair_batch_on=None,
    **kwargs
):

    if isinstance(return_as, str):
        return_as = [return_as]

    assert set(return_as).issubset({"anndata", "dataset", "loader"})
    config.data.condition = config.data.get("condition", "drug")
    condition = config.data.condition

    if data is None:
        if config.data.type == "cell":
            data = read_single_anndata(config, **kwargs)
        else:
            raise ValueError

    if config.data.get("select") is not None:
        keep = pd.Series(False, index=data.obs_names)
        for key, value in config.data.select.items():
            if not isinstance(value, list):
                value = [value]
            keep.loc[data.obs[key].isin(value)] = True
            assert keep.sum() > 0

        data = data[keep].copy()

    if "dimension_reduction" in config.data:
        genes = data.var_names.to_list()
        name = config.data.dimension_reduction.name
        if name == "pca":
            dims = config.data.dimension_reduction.get(
                "dims", data.obsm["X_pca"].shape[1]
            )

            data = anndata.AnnData(
                data.obsm["X_pca"][:, :dims], obs=data.obs.copy(), uns=data.uns.copy()
            )
            data.uns["genes"] = genes

    if "ae_emb" in config.data:
        # load path to autoencoder
        assert config.get("model.name", "cellot") == "cellot"
        path_ae = Path(config.data.ae_emb.path)
        model_kwargs = {"input_dim": data.n_vars}
        config_ae = load_config(path_ae / "config.yaml")
        ae_model, _ = load_autoencoder_model(
            config_ae, restore=path_ae / "cache/model.pt", **model_kwargs
        )

        inputs = torch.Tensor(
            data.X if not sparse.issparse(data.X) else data.X.todense()
        )

        genes = data.var_names.to_list()
        data = anndata.AnnData(
            ae_model.eval().encode(inputs).detach().numpy(),
            obs=data.obs.copy(),
            uns=data.uns.copy(),
        )
        data.uns["genes"] = genes

    # cast to dense and check for nans
    if sparse.issparse(data.X):
        data.X = data.X.todense()
    assert not np.isnan(data.X).any()

    dataset_args = dict()
    model_kwargs = {}

    model_kwargs["input_dim"] = data.n_vars

    if config.get("model.name") == "cae":
        condition_labels = sorted(data.obs[condition].cat.categories)
        model_kwargs["conditions"] = condition_labels
        dataset_args["obs"] = condition
        dataset_args["categories"] = condition_labels

    if "training" in config:
        pair_batch_on = config.training.get("pair_batch_on", pair_batch_on)

    if split_on is None:
        if config.model.name == "cellot":
            # datasets & dataloaders accessed as loader.train.source
            split_on = ["split", "transport"]
            if pair_batch_on is not None:
                split_on.append(pair_batch_on)

        elif (
            config.model.name == "scgen"
            or config.model.name == "cae"
            or config.model.name == "popalign"
        ):
            split_on = ["split"]

        else:
            raise ValueError

    if isinstance(split_on, str):
        split_on = [split_on]

    for key in split_on:
        assert key in data.obs.columns

    if len(split_on) > 0:
        splits = {
            (key if isinstance(key, str) else ".".join(key)): data[index]
            for key, index in data.obs[split_on].groupby(split_on).groups.items()
        }

        dataset = nest_dict(
            {
                key: AnnDataDataset(val.copy(), **dataset_args)
                for key, val in splits.items()
            },
            as_dot_dict=True,
        )

    else:
        dataset = AnnDataDataset(data.copy(), **dataset_args)

    if "loader" in return_as:
        kwargs = dict(config.dataloader)
        kwargs.setdefault("drop_last", True)
        loader = cast_dataset_to_loader(dataset, **kwargs)

    returns = list()
    for key in return_as:
        if key == "anndata":
            returns.append(data)

        elif key == "dataset":
            returns.append(dataset)

        elif key == "loader":
            returns.append(loader)

    if include_model_kwargs:
        returns.append(model_kwargs)

    if len(returns) == 1:
        return returns[0]

    return tuple(returns)
