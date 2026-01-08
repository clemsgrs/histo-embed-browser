import torch
import numpy as np
import marimo as mo
from pathlib import Path
from collections import defaultdict


def load_features_and_metadata(
    *,
    df,
    device,
    **kwargs,
):
    features_list = []
    tile_indices_list = []
    wsi_paths_list = []
    coordinates_paths_list = []
    metadata = defaultdict(list)
    metadata_cols = ["case_id"] + [x for x in df.columns if x not in ["wsi_path", "feature_path", "coordinates_path"]]

    num_tiles_per_wsi = kwargs.get("num_tiles_per_wsi", 1)
    g = torch.Generator(device=device).manual_seed(0)

    iterator = mo.status.progress_bar(
        df.itertuples(index=False),
        total=len(df),
        title="Reading embeddings files...",
        completion_title="Embeddings loaded.",
        show_eta=True,
        show_rate=True,
    )
    for row in iterator:
        name = Path(row.wsi_path).stem
        _feature = torch.load(row.feature_path, map_location=device)
        sampled_indices = torch.randperm(len(_feature), generator=g, device=device)[:num_tiles_per_wsi]
        sampled_feature = _feature[sampled_indices].clone()
        features_list.append(sampled_feature.cpu())
        wsi_paths_list.extend([row.wsi_path]*len(sampled_indices))
        coordinates_paths_list.extend([row.coordinates_path]*len(sampled_indices))
        tile_indices_list.extend(list(sampled_indices.cpu().numpy()))
        for col_name in metadata_cols:
            if col_name == "case_id":
                metadata["case_id"].extend([name]*len(sampled_indices))
            else:
                val = getattr(row, col_name)
                metadata[col_name].extend([val]*len(sampled_indices))
    
    features = np.concatenate(features_list)
    return {
        "features": features,
        "tile_indices": tile_indices_list,
        "wsi_paths": wsi_paths_list,
        "coordinates_paths": coordinates_paths_list,
        "metadata": metadata,
        "metadata_cols": metadata_cols,
    }
