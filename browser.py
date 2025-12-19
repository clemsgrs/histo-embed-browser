import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    <h1 style="text-align: center;">Histo Embed Browser</h1>
    """)
    return


@app.cell
def _():
    import torch
    import pandas as pd
    import marimo as mo
    import altair as alt

    from sklearn.manifold import TSNE

    from src.utils import get_cfg_from_file, load_features_and_metadata, scatter, clickable_image_preview
    return (
        TSNE,
        alt,
        clickable_image_preview,
        get_cfg_from_file,
        load_features_and_metadata,
        mo,
        pd,
        scatter,
        torch,
    )


@app.cell
def _(mo):
    # increase max bytes to allow display of more thumbails
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10000000000
    return


@app.cell
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip
        await micropip.install("altair")
    return


@app.cell
def _():
    # Load input
    return


@app.cell
def _(get_cfg_from_file, pd):
    cfg = get_cfg_from_file("config.yaml")
    df = pd.read_csv(cfg.csv)
    return cfg, df


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _(cfg):
    kwargs = {"num_tiles_per_wsi": cfg.options.num_tiles_per_wsi}
    return (kwargs,)


@app.cell
def _(device, df, kwargs, load_features_and_metadata, mo):
    with mo.status.spinner(title="Loading embeddings...", subtitle="This can take a moment") as _sp:
        out = load_features_and_metadata(
            df=df,
            device=device,
            **kwargs,
        )
        _sp.update("Embeddings loaded")
    return (out,)


@app.cell
def _(TSNE, out, mo):
    with mo.status.spinner(title="Computing t-SNE…", subtitle="Please wait...") as _sp:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
        emb2d = tsne.fit_transform(out["features"])
        _sp.update("t-SNE computation complete")
    return (emb2d,)


@app.cell
def _(emb2d, out, pd):
    tsne_df = pd.DataFrame(emb2d, columns=["x", "y"]).reset_index()

    metadata_df = pd.DataFrame({
        m: l
        for m, l in out["metadata"].items()
    })

    full_df = pd.concat([tsne_df.reset_index(drop=True), metadata_df], axis=1)
    full_df["tile_idx"] = out["tile_indices"]
    full_df["wsi_path"] = out["wsi_paths"]
    full_df["coordinates_path"] = out["coordinates_paths"]
    return (full_df,)


@app.cell(hide_code=True)
def _(mo, emb2d):
    mo.stop(emb2d is None)
    mo.md(r"""
    Choose color scheme
    """)
    return


@app.cell
def _(mo, out, emb2d):
    mo.stop(emb2d is None)
    label = mo.ui.dropdown(out["metadata_cols"], value="case_id")
    label
    return (label,)


@app.cell
def _(alt, label, out):
    label_to_color = {
        m: alt.Color(f"{label.value}:N")
        for m in out["metadata_cols"]
    }
    return (label_to_color,)


@app.cell
def _(full_df, label, label_to_color, mo, scatter):
    chart = mo.ui.altair_chart(
        scatter(
            df=full_df,
            color=label_to_color[label.value]
        )
    )
    chart
    return (chart,)


@app.cell
def _(chart, mo):
    table = mo.ui.table(chart.value)
    return (table,)


@app.cell(hide_code=True)
def _(cfg, chart, clickable_image_preview, full_df, mo, table):
    mo.stop(not len(chart.value))

    selected_indices = (
        list(chart.value["index"])
        if not len(table.value)
        else list(table.value["index"])
    )

    with mo.status.spinner(
        title="Loading image previews…",
        subtitle=f"Sampling from {len(selected_indices)} images"
    ) as _sp:
        gallery = clickable_image_preview(
            df=full_df,
            indices=selected_indices, 
            context_dim=cfg.options.context_dim,
            )
        _sp.update("Previews ready.")

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:

        {gallery}

        Here's all the data you've selected.

        {table}
        """
    )
    return


if __name__ == "__main__":
    app.run()
