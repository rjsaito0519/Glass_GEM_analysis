# scripts

- **`modules/common.py`** — CSV I/O、run レイアウト、窓・スパーク、PSD、しきい値交差など共通処理。
- **`modules/event_catalog.py`** — 波形イベントカタログ（1 行 1 GGEM CSV）の生成と **`main()`**。FFT は分類ラベル用の一部。
- **`build_event_catalog.py`** — `PYTHONPATH=scripts` で実行。

設定 JSON はリポジトリ直下の **`conf/`**（`--conf` 省略時は `conf/event_catalog.default.json`）。

```bash
cd /path/to/analysis
PYTHONPATH=scripts python3 scripts/build_event_catalog.py /path/to/run1
PYTHONPATH=scripts python3 scripts/build_event_catalog.py --conf conf/my.json /path/to/run1 /path/to/run2
```

インストール時はコンソールスクリプト **`build-event-catalog`** でも起動可（`pyproject.toml`）。run は必ず末尾の **`RUN_DIR`** で渡す（conf には書かない）。

詳細は [conf/README.md](../conf/README.md)（リポジトリ直下の `conf/`）。

旧ツリーは `backup/legacy_20260513/scripts/` を参照。
