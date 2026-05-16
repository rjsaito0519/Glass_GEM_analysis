# conf — イベントカタログ

- **`event_catalog.default.json`** — `--conf` 省略時に読む設定（実験ごとに変えたい項目のみ記載）。
- 未記載のキーは `scripts/modules/event_catalog.py` の `_DEFAULT_CATALOG_JSON` で補完される。

`RUN_DIR` は JSON には書かず、CLI 末尾に 1 個以上渡す。

## 既定 JSON に含める項目

| キー | 意味 |
|------|------|
| `ggem_channel_id` | 主波形 `CH{n}/csv/` |
| `nim_channel_id` | NIM 波形 `CH{n}/csv/` |
| `tmin_us` / `tmax_us` | 解析窓 [µs] |
| `spark_threshold_mv` / `spark_min_duration_us` | スパーク判定 |
| `noise_dominant_peak_min_mhz` | FFT ノイズ判定のピークしきい [MHz] |
| `edge_mv` / `nim_width_ns` | NIM 立下り・立上り推定 |

## コード側の既定のみ（必要なら `--conf` で上書き可）

`output_root`（省略時 `analysis/results/`）、`csv_subdir`、`rel_tol_dt`、`subtract_mean_fft`、`timestamp_regex`、`timestamp_format`、`baseline_png`

```bash
cd /path/to/analysis
PYTHONPATH=scripts python3 scripts/decode_event_catalog.py /path/to/run1
PYTHONPATH=scripts python3 scripts/decode_event_catalog.py --conf conf/my.json /path/to/run1
```
