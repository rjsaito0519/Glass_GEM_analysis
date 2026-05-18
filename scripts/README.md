# scripts

- **`modules/common.py`** — CSV I/O、run レイアウト、窓・スパーク、PSD、しきい値交差など共通処理。
- **`modules/event_catalog.py`** — 波形イベントカタログ（1 行 1 GGEM CSV）の生成と **`main()`**。FFT は分類ラベル用の一部。
- **`decode_event_catalog.py`** — `PYTHONPATH=scripts` で実行。
- **`check_waveform.py`** — 全 CH 波形の表示のみ。`n` / `p` / `q`、``--n N``、``--bcor``（GGEM のみ decode ベースライン減算）。
- **`plot_event_catalog.py`** — デコード済み `event_catalog.csv` から integral / vmax ヒストグラム。``--same`` で run 重ね（signal のみ、density 正規化）。
- **`merge_event_catalog.py`** — 複数の `event_catalog.csv`（または `results/runXXXX/`）を 1 本に結合。``--out NAME`` で `results/<NAME>/` に出力。

設定 JSON はリポジトリ直下の **`conf/`**（`--conf` 省略時は `conf/event_catalog.default.json`）。

```bash
cd /path/to/analysis
PYTHONPATH=scripts python3 scripts/decode_event_catalog.py /path/to/run1
PYTHONPATH=scripts python3 scripts/decode_event_catalog.py --conf conf/my.json /path/to/run1 /path/to/run2
```

```bash
PYTHONPATH=scripts python3 scripts/check_waveform.py /path/to/run
PYTHONPATH=scripts python3 scripts/check_waveform.py --n 5 /path/to/run
PYTHONPATH=scripts python3 scripts/check_waveform.py --bcor /path/to/run0093
PYTHONPATH=scripts python3 scripts/plot_event_catalog.py results/run0089 results/run0091
PYTHONPATH=scripts python3 scripts/plot_event_catalog.py --same results/run0089 results/run0091
PYTHONPATH=scripts python3 scripts/merge_event_catalog.py results/run0089 results/run0091 --out merged_0089_0091
```

インストール時はコンソールスクリプト **`decode-event-catalog`** / **`check-waveform`** / **`plot-event-catalog`** / **`merge-event-catalog`** でも起動可（`pyproject.toml`）。`decode_event_catalog` の run は末尾の **`RUN_DIR`**（conf には書かない）。

詳細は [conf/README.md](../conf/README.md)（リポジトリ直下の `conf/`）。

旧ツリーは `backup/legacy_20260513/scripts/` を参照。
