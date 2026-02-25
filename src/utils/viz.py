import altair as alt
import numpy as np
import wholeslidedata as wsd
from PIL import Image, ImageDraw


def get_tile(
    *,
    wsi_path,
    coordinates,
    tile_idx,
    context_dim: int = 0,
):
    wsi = wsd.WholeSlideImage(wsi_path, backend="asap")
    tile_level = coordinates["tile_level"][tile_idx]
    tile_spacing = wsi.spacings[tile_level]
    tile_size_resized = coordinates["tile_size_resized"][tile_idx]
    x, y = coordinates["x"][tile_idx], coordinates["y"][tile_idx]
    tile_arr = wsi.get_patch(
        x,
        y,
        tile_size_resized,
        tile_size_resized,
        spacing=tile_spacing,
        center=False,
    )
    tile = Image.fromarray(tile_arr).convert("RGB")
    # potential resize tile to account for difference in tile spacing and target spacing
    resize_factor = coordinates["resize_factor"][tile_idx]
    tile_size = int(round(tile_size_resized / resize_factor, 0))
    if tile_size != tile_size_resized:
        tile = tile.resize((tile_size, tile_size))
    if context_dim > 0:
        tile_size_at_0 = coordinates["tile_size_at_0"][tile_idx]
        # context_dim controls number of surrounding tiles to fetch
        num_tiles_in_context = 2 * context_dim + 1
        x_shifted = x - context_dim * tile_size_at_0
        y_shifted = y - context_dim * tile_size_at_0
        width = tile_size_resized * num_tiles_in_context
        height = tile_size_resized * num_tiles_in_context
        ctx_tile_arr = wsi.get_patch(
            x_shifted,
            y_shifted,
            width,
            height,
            spacing=tile_spacing,
            center=False,
        )
        ctx_tile = Image.fromarray(ctx_tile_arr).convert("RGB")
        if tile_size != tile_size_resized:
            w = tile_size * num_tiles_in_context
            h = tile_size * num_tiles_in_context
            ctx_tile = ctx_tile.resize((w, h))
        ctx_tile = dim_and_draw_border(ctx_tile, tile_size, context_dim=context_dim)
    else:
        ctx_tile = None
    return {"tile": tile, "ctx_tile": ctx_tile}


def dim_and_draw_border(
    *,
    img: Image,
    tile_size: int,
    overlay_alpha: float = 0.5,
    border_color=(0, 0, 0),
    border_width: int = 4,
):
    w, h = img.size
    left = (w - tile_size) // 2
    top = (h - tile_size) // 2
    right = left + tile_size
    bottom = top + tile_size
    center_box = (left, top, right, bottom)

    # convert to RGBA
    base = img.convert("RGBA")

    # create semi-transparent overlay
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # fill whole overlay with semi-transparent black
    draw.rectangle([0, 0, w, h], fill=(0, 0, 0, int(255 * overlay_alpha)))
    # make the center box transparent (cut-out)
    draw.rectangle(center_box, fill=(0, 0, 0, 0))

    # composite: overlay on top of base
    combined = Image.alpha_composite(base, overlay)

    # draw colored border on top (use RGBA so alpha is preserved)
    draw2 = ImageDraw.Draw(combined)
    # PIL draws border inside the rectangle; to get a symmetric border you may expand by half width
    outer = (
        left - border_width // 2,
        top - border_width // 2,
        right + border_width // 2,
        bottom + border_width // 2,
    )
    draw2.rectangle(outer, outline=border_color + (255,), width=border_width)

    return combined.convert("RGB")


def scatter(
    *,
    df,
    color,
):
    return (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x="x",
            y="y",
            color=color,
        )
        .properties(width=500, height=500)
    )


def clickable_image_preview(
    *,
    df,
    indices,
    context_dim: int = 0,
    max_images: int = 20,
):
    import base64
    import uuid
    import io

    np.random.seed(21)
    indices = np.array(indices)
    np.random.shuffle(indices)
    indices = indices[:max_images]

    html_parts = []

    # CSS-only lightbox implementation using the "checkbox hack"
    style = """
    <style>
        .gallery-item {
            display: inline-block;
            margin: 2px;
        }
        .thumbnail {
            height: 100px;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.1s;
        }
        .thumbnail:hover {
            transform: scale(1.05);
        }

        /* Lightbox hidden by default */
        .lightbox {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 10000;
            justify-content: center;
            align-items: center;
        }

        /* Show lightbox when checkbox is checked */
        .lightbox-toggle:checked + .thumb-label + .lightbox {
            display: flex;
        }

        /* Overlay background - clicking this closes the modal */
        .lightbox-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            cursor: zoom-out;
        }

        .full-image {
            max-width: 90vw;
            max-height: 90vh;
            z-index: 10001;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
            border: 2px solid white;
        }
    </style>
    """
    html_parts.append(style)
    html_parts.append('<div style="display: flex; flex-wrap: wrap;">')

    for idx in indices:
        tile_idx = df.tile_idx.loc[idx]
        wsi_path = df.wsi_path.loc[idx]
        coordinates_path = df.coordinates_path.loc[idx]
        coordinates = np.load(coordinates_path, allow_pickle=True)
        res = get_tile(
            wsi_path=wsi_path,
            coordinates=coordinates,
            tile_idx=tile_idx,
            context_dim=context_dim,
        )
        thumb = res["tile"]
        thumb.thumbnail((150, 150))
        thumb_buffer = io.BytesIO()
        thumb.save(thumb_buffer, format="PNG")
        thumb_b64 = base64.b64encode(thumb_buffer.getvalue()).decode("utf-8")
        thumb_uri = f"data:image/png;base64,{thumb_b64}"

        if context_dim > 0:
            full = res["ctx_tile"]
            full_buffer = io.BytesIO()
            full.save(full_buffer, format="PNG")
            full_b64 = base64.b64encode(full_buffer.getvalue()).decode("utf-8")
            full_uri = f"data:image/png;base64,{full_b64}"
        else:
            full_uri = thumb_uri

        uid = str(uuid.uuid4())

        item_html = f"""
        <div class="gallery-item">
            <input type="checkbox" id="cb-{uid}" class="lightbox-toggle" style="display: none;">

            <label for="cb-{uid}" class="thumb-label">
                <img src="{thumb_uri}" class="thumbnail" />
            </label>

            <div class="lightbox">
                <label for="cb-{uid}" class="lightbox-overlay"></label>
                <img src="{full_uri}" class="full-image" />
            </div>
        </div>
        """
        html_parts.append(item_html)

    html_parts.append("</div>")
    return "".join(html_parts)
