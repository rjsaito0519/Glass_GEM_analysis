# scripts

- **`modules/common.py`** — CSV I/O、run レイアウト、窓・スパーク、PSD、しきい値交差など共通処理。
- **`modules/event_catalog.py`** — 波形イベントカタログ（1 行 1 GGEM CSV）の生成と **`main()`**。FFT は分類ラベル用の一部。
- **`decode_event_catalog.py`** — `PYTHONPATH=scripts` で実行。
- **`check_waveform.py`** — 全 CH 波形の表示のみ。`n` / `p` / `q`、``--n N``、``--bcor``（GGEM のみ decode ベースライン減算）。
- **`plot_integ_maxv.py`** — デコード済み `event_catalog.csv` から integral / vmax ヒストグラム。``--same`` で **PATH（results サブディレクトリ）ごと**に重ね（signal のみ、density 正規化、凡例=ディレクトリ名）。
- **`plot_integ_maxv_corr.py`** — 同一イベントの integral–vmax 2 次元ヒスト（矩形メッシュ、1D hist と同じ bin）。signal / noise を別パネル表示。
- **`plot_dt_ggem_nim.py`** — `dt_ggem_nim_ns` ヒスト。右クリック 2 回で範囲指定、``w`` でその範囲の integ/max 追従表示。
- **`plot_trend_sn_spark.py`** — 時間ビンごとの signal 件数・S/N・スパーク率 [/10 min]・**baseline_indiv_mv 平均**（非スパークのみ、``--span`` 分幅、既定 10）。segmented 時は run 全体 ``baseline_mv`` を破線で重ねる。PATH 1 個のみ exp fit。複数 PATH で **1 時間未満**しか空かないときは全行まとめて binning（実時間軸）；**1 時間以上**の空きがあるときだけ run 別 binning＋境界破線＋ギャップ圧縮（区切り約 60 分）。
- **`merge_event_catalog.py`** — 複数の `event_catalog.csv`（または `results/runXXXX/`）を 1 本に結合。``--out NAME`` で `results/<NAME>/` に出力。
- **`summarize_runs.py`** — 引数なしで `results/run*/event_catalog.csv` を探索し、signal/noise/spark 件数・測定時間・spark/h・S/N・signal 平均 integ/maxv の表を print。

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
PYTHONPATH=scripts python3 scripts/plot_integ_maxv.py results/run0089 results/run0091
PYTHONPATH=scripts python3 scripts/plot_integ_maxv.py --same results/295_1st results/run0100
PYTHONPATH=scripts python3 scripts/plot_integ_maxv.py --same results/run0089 results/run0091
PYTHONPATH=scripts python3 scripts/plot_integ_maxv_corr.py results/run0093
PYTHONPATH=scripts python3 scripts/plot_integ_maxv_corr.py results/run0097 results/run0098
PYTHONPATH=scripts python3 scripts/plot_dt_ggem_nim.py results/run0089 --dt-auto
PYTHONPATH=scripts python3 scripts/plot_trend_sn_spark.py results/run0089 --span 10
PYTHONPATH=scripts python3 scripts/plot_trend_sn_spark.py results/run0089 results/run0091 --span 10
PYTHONPATH=scripts python3 scripts/plot_trend_sn_spark.py results/run0089 results/run0091 --no-compress-gaps
PYTHONPATH=scripts python3 scripts/merge_event_catalog.py results/run0089 results/run0091 --out merged_0089_0091
PYTHONPATH=scripts python3 scripts/summarize_runs.py
```

インストール時はコンソールスクリプト **`decode-event-catalog`** / **`check-waveform`** / **`plot-integ-maxv`** / **`plot-integ-maxv-corr`** / **`plot-dt-ggem-nim`** / **`plot-trend-sn-spark`** / **`merge-event-catalog`** / **`summarize-runs`** でも起動可（`pyproject.toml`）。`decode_event_catalog` の run は末尾の **`RUN_DIR`**（conf には書かない）。

詳細は [conf/README.md](../conf/README.md)（リポジトリ直下の `conf/`）。

旧ツリーは `backup/legacy_20260513/scripts/` を参照。
