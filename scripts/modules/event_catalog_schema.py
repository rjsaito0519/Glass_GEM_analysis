"""イベントカタログ CSV の列名定義。"""

CATALOG_CSV_COLUMNS = (
    "run_num",         # この行が属する run の識別子（通常は run ディレクトリ名など）
    "ggem_csv",        # 主波形（GGEM）の CSV ファイル名（stem ではなくファイル名）
    "timestamp",       # 設定の正規表現・日時書式からファイル名に埋め込まれた時刻（失敗時は空文字）
    "file_index",      # 当 run 内でソートされた GGEM CSV 列の 0 始まりインデックス
    "spark",           # 全レンジ上でスパークと判定されれば 1、そうでなければ 0
    "fft_is_noise",    # スパーク時は空。非スパークのみ: PSD ノイズ判定なら "1"、信号側 "0"、判定不能 ""
    "baseline_mv",     # run 全体: 窓外平均プールのヒストに lmfit GaussianModel の center [mV]（失敗時は算術平均）。窓内から減算
    "integral_mv_us",  # 解析窓内、(v − baseline) の台形積分（単位は mV·µs）
    "vmax_mv",         # 解析窓内、(v − baseline) の最大値 [mV]
    "dt_ggem_nim_ns",  # スパーク時は空。非スパーク: GGEM 窓内最大時刻と NIM 落下推定時刻の差 [ns]（符号は設定）
)
