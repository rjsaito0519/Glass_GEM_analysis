# scripts

- **`modules/common.py`** — CSV I/O、run レイアウト、窓・スパーク、PSD、しきい値交差など共通処理。
- **`modules/event_catalog.py`** — 波形イベントカタログ（1 行 1 GGEM CSV）の生成と **`main()`**。FFT は分類ラベル用の一部。
- **`decode_event_catalog.py`** — `PYTHONPATH=scripts` で実行。
- **`check_waveform.py`** — 全 CH 波形の表示のみ。`n` / `p` / `q`、``--n N`` で重ね描画（N>1 はバッチ送り）。

設定 JSON はリポジトリ直下の **`conf/`**（`--conf` 省略時は `conf/event_catalog.default.json`）。

```bash
cd /path/to/analysis
PYTHONPATH=scripts python3 scripts/decode_event_catalog.py /path/to/run1
PYTHONPATH=scripts python3 scripts/decode_event_catalog.py --conf conf/my.json /path/to/run1 /path/to/run2
```

```bash
PYTHONPATH=scripts python3 scripts/check_waveform.py /path/to/run
PYTHONPATH=scripts python3 scripts/check_waveform.py --n 5 /path/to/run
```

インストール時はコンソールスクリプト **`decode-event-catalog`** / **`check-waveform`** でも起動可（`pyproject.toml`）。`decode_event_catalog` の run は末尾の **`RUN_DIR`**（conf には書かない）。

詳細は [conf/README.md](../conf/README.md)（リポジトリ直下の `conf/`）。

旧ツリーは `backup/legacy_20260513/scripts/` を参照。
