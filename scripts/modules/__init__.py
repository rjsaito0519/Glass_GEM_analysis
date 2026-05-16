"""Public exports for ``PYTHONPATH=scripts`` workflows."""

from modules.cli_common import STANDARD_CLI_EPILOG, add_standard_cli_arguments
from modules.common import (
    aligned_csv_paths,
    discover_channel_roots,
    dt_us,
    fft_is_noise_label,
    is_spark_event,
    one_sided_psd_m2_per_hz,
    read_waveform_csv,
    sorted_waveform_csvs,
    time_and_max,
    window_arrays,
)
from modules.event_catalog import (
    CATALOG_CSV_COLUMNS,
    EventCatalogRow,
    default_catalog_config_path,
    load_catalog_build_config,
    process_run_to_rows,
    unique_result_subdir,
    write_catalog,
)

__all__ = [
    "add_standard_cli_arguments",
    "aligned_csv_paths",
    "CATALOG_CSV_COLUMNS",
    "default_catalog_config_path",
    "discover_channel_roots",
    "dt_us",
    "EventCatalogRow",
    "fft_is_noise_label",
    "is_spark_event",
    "load_catalog_build_config",
    "one_sided_psd_m2_per_hz",
    "process_run_to_rows",
    "read_waveform_csv",
    "sorted_waveform_csvs",
    "STANDARD_CLI_EPILOG",
    "time_and_max",
    "unique_result_subdir",
    "window_arrays",
    "write_catalog",
]
