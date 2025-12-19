# histo-embed-browser
Interactive tSNE browser for histopathology embeddings with on-demand image inspection

## 🛠️ Installation

```bash
pip3 install -r requirements.in
```

## 🚀 Usage

### 1. 📊 Data Preparation
You need a **csv** file with at least the following columns:

- `wsi_path`: path to whole slide images
- `feature_path`: path to pre-extracted features (`.pt` file)

Features can be 1D (slide-level) or 2D (tile-level).
For tile-level features, the **csv** must also include:

- `coordinates_path`: path to the tile coordinates (`.npy` file)

> **Tip:** You can generate compatible features and coordinates using the [slide2vec](https://github.com/clemsgrs/slide2vec/tree/main) repository.

The **csv** may also include any additional slide-level metadata (e.g. labels) that will be available for coloring the points in the tSNE visualization.

> **Note:** Currently, only categorical (discrete) values are supported. A future update will allow continuous values.

### 2. ⚙️ Configuration

Edit `config.yaml` to point to your data:

1. Set `csv` to the path of your csv file.
2. If using tile-level features, set `level` to `tile` in the configuration options:
   ```yaml
   options:
     level: "tile"
   ```

### 3. 🖥️ Run the Browser

Start the interactive tSNE browser using marimo:

```bash
marimo run browser.py
```
