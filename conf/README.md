# conf — イベントカタログ用 JSON

- **`event_catalog.default.json`** — `--conf` を付けずに実行したときに読むファイル。`run_dirs` を自分の環境用に編集する。
- **`event_catalog.example.json`** — コピー用の同内容サンプル。

未記載のキーは `modules/event_catalog.py` の `_DEFAULT_CATALOG_JSON` で補完される。

主なキー（用語）:

- **`ggem_channel_id`** — カタログの主波形（従来の CH1 相当）。`CH{番号}/csv/` を参照。
- **`nim_channel_id`** — NIM ロジック側波形（従来の CH2 相当）。`dt_ggem_nim_ns` の落下時刻に使用。
- **`dt_nim_fall_minus_ggem_peak`** — `true` のとき Δt は「NIM 落下 − GGEM 窓内最大時刻」方向（ns）。旧キー `dt_fall_minus_ch1_peak` も読み込み時に引き継がれる。

出力 CSV の主な列: `run_num`（run 識別子）、`ggem_csv`、`baseline_mv`（窓内減算に使う run 全体ベースライン [mV]）、`dt_ggem_nim_ns` など。成果物のファイル名は `event_catalog.csv` / `event_catalog.meta.json` / `event_catalog_baseline.png`（既定）。

```bash
cd /path/to/analysis
PYTHONPATH=scripts python3 scripts/build_event_catalog.py
PYTHONPATH=scripts python3 scripts/build_event_catalog.py --conf conf/my_runs.json
```

- `run_dirs` の相対パスは **その JSON ファイルがあるディレクトリ** を基準に解決する。
- `output_root` が `null` または省略なら `analysis/results/`。
