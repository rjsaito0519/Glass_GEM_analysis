"""CSV column names for the event catalog."""

CATALOG_CSV_COLUMNS = (
    "run_num",           # run 識別子
    "ggem_csv",          # GGEM 波形 CSV ファイル名
    "timestamp",         # ファイル名から取った時刻（失敗時は空）
    "file_index",        # run 内 GGEM 列の 0 始まりインデックス
    "spark",             # スパークなら 1
    "fft_is_noise",      # 非スパーク: "1"/"0"、スパーク・不明は空
    "baseline_mv",       # run 全体ベースライン [mV]（窓内減算に使用）
    "baseline_indiv_mv", # 当該 CSV の窓外平均 [mV]（窓外なしは空）
    "integral_mv_us",    # 窓内 (v − baseline_mv) の積分 [mV·µs]
    "vmax_mv",           # 窓内 (v − baseline_mv) の最大 [mV]
    "dt_ggem_nim_ns",    # 非スパーク: GGEM 窓内最大と NIM 落下の差 [ns]
)
