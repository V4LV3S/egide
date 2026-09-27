# from __future__ import annotations
#
# import json
# import logging
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import Iterable
#
# import numpy as np
# import pandas as pd
#
#
# # ============================================================
# # CONFIG
# # ============================================================
#
# RAW_DIR = Path("data/raw")
# PROCESSED_DIR = Path("data/processed")
# TRAIN_DIR = Path("data/training")
# SEQUENCE_DIR = TRAIN_DIR / "lstm_cnn"
#
# STATE_HOURLY_PATH = PROCESSED_DIR / "state_hourly.parquet"
# FEATURE_COLUMNS_PATH = TRAIN_DIR / "feature_columns.json"
# STATE_MAPPING_PATH = TRAIN_DIR / "state_mapping.json"
# COVERAGE_REPORT_PATH = TRAIN_DIR / "coverage_report.csv"
#
# X_PATH = SEQUENCE_DIR / "X.npy"
# Y_FLAG_PATH = SEQUENCE_DIR / "y_curtailment_flag.npy"
# Y_MW_PATH = SEQUENCE_DIR / "y_curtailment_mwmed.npy"
# Y_MWH_PATH = SEQUENCE_DIR / "y_curtailment_mwh.npy"
# Y_MINUTES_PATH = SEQUENCE_DIR / "y_curtailment_minutes.npy"
# Y_REASON_PATH = SEQUENCE_DIR / "y_reason.npy"
# STATE_IDX_PATH = SEQUENCE_DIR / "state_idx.npy"
# END_DATETIME_PATH = SEQUENCE_DIR / "end_datetime.npy"
#
# EOLIC_DIR = RAW_DIR / "constrained_off_eolico"
# LOAD_DIR = RAW_DIR / "curva_carga"
# INTERCHANGE_DIR = RAW_DIR / "intercambio_subsistemas"
#
# EXPECTED_REASON_COLS = [
#     "reason_REL",
#     "reason_CNF",
#     "reason_ENE",
#     "reason_PAR",
# ]
#
# EXPECTED_SEQUENCE_LENGTH = 168
# EXPECTED_REASON_COUNT = 4
#
# # BA_SE é deliberadamente uma unidade espacial do modelo.
# # Estados sem geração eólica não são obrigados a aparecer.
# EXPECTED_MODEL_STATES = {
#     "BA_SE",
#     "CE",
#     "PB",
#     "PE",
#     "PI",
#     "RN",
# }
#
# CSV_KW = {
#     "sep": ";",
#     "encoding": "utf-8-sig",
# }
#
# CHUNKSIZE = 500_000
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
# )
#
# logger = logging.getLogger("datacheck")
#
#
# # ============================================================
# # RESULTADO DO CHECK
# # ============================================================
#
# @dataclass
# class CheckResult:
#     check: str
#     status: str
#     message: str
#     value: object | None = None
#
#
# RESULTS: list[CheckResult] = []
#
#
# def ok(
#     check: str,
#     message: str,
#     value: object | None = None,
# ) -> None:
#     RESULTS.append(
#         CheckResult(
#             check=check,
#             status="OK",
#             message=message,
#             value=value,
#         )
#     )
#
#
# def warning(
#     check: str,
#     message: str,
#     value: object | None = None,
# ) -> None:
#     RESULTS.append(
#         CheckResult(
#             check=check,
#             status="WARNING",
#             message=message,
#             value=value,
#         )
#     )
#
#
# def error(
#     check: str,
#     message: str,
#     value: object | None = None,
# ) -> None:
#     RESULTS.append(
#         CheckResult(
#             check=check,
#             status="ERROR",
#             message=message,
#             value=value,
#         )
#     )
#
#
# # ============================================================
# # HELPERS
# # ============================================================
#
# def read_csv_header(path: Path) -> list[str]:
#     return pd.read_csv(
#         path,
#         nrows=0,
#         **CSV_KW,
#     ).columns.tolist()
#
#
# def list_csvs(directory: Path) -> list[Path]:
#     files = sorted(
#         directory.rglob("*.csv")
#     )
#
#     if not files:
#         raise FileNotFoundError(
#             f"Nenhum CSV encontrado em {directory}"
#         )
#
#     return files
#
#
# def parse_datetime(
#     series: pd.Series,
# ) -> pd.Series:
#     return (
#         pd.to_datetime(
#             series,
#             errors="coerce",
#         )
#         .dt
#         .tz_localize(None)
#     )
#
#
# def date_range_stats(
#     dates: pd.Series,
#     freq: str,
# ) -> dict:
#     dates = dates.dropna()
#
#     if dates.empty:
#         return {
#             "min": None,
#             "max": None,
#             "n_unique": 0,
#             "n_gaps": None,
#             "max_gap_hours": None,
#         }
#
#     unique_dates = (
#         pd.Series(
#             dates.unique()
#         )
#         .sort_values()
#         .reset_index(drop=True)
#     )
#
#     diff = unique_dates.diff()
#
#     expected_delta = pd.Timedelta(
#         hours=1
#     )
#
#     if freq == "30min":
#         expected_delta = pd.Timedelta(
#             minutes=30
#         )
#
#     gaps = diff[
#         diff > expected_delta
#     ]
#
#     max_gap = (
#         gaps.max()
#         if not gaps.empty
#         else pd.Timedelta(0)
#     )
#
#     return {
#         "min": unique_dates.min(),
#         "max": unique_dates.max(),
#         "n_unique": len(unique_dates),
#         "n_gaps": len(gaps),
#         "max_gap_hours": (
#             max_gap.total_seconds() / 3600
#             if max_gap
#             else 0
#         ),
#     }
#
#
# def print_summary() -> None:
#     print("\n" + "=" * 90)
#     print("DATA CHECK SUMMARY")
#     print("=" * 90)
#
#     for result in RESULTS:
#         value_text = (
#             f" | {result.value}"
#             if result.value is not None
#             else ""
#         )
#
#         print(
#             f"[{result.status:7}] "
#             f"{result.check}: "
#             f"{result.message}"
#             f"{value_text}"
#         )
#
#     errors = sum(
#         result.status == "ERROR"
#         for result in RESULTS
#     )
#
#     warnings = sum(
#         result.status == "WARNING"
#         for result in RESULTS
#     )
#
#     print("-" * 90)
#     print(
#         f"ERRORS={errors} | WARNINGS={warnings}"
#     )
#
#
# # ============================================================
# # CHECK 1 — ARQUIVOS
# # ============================================================
#
# def check_required_files() -> bool:
#     required = [
#         STATE_HOURLY_PATH,
#         FEATURE_COLUMNS_PATH,
#         STATE_MAPPING_PATH,
#         X_PATH,
#         Y_FLAG_PATH,
#         Y_MW_PATH,
#         Y_MWH_PATH,
#         Y_MINUTES_PATH,
#         Y_REASON_PATH,
#         STATE_IDX_PATH,
#         END_DATETIME_PATH,
#     ]
#
#     missing = [
#         str(path)
#         for path in required
#         if not path.exists()
#     ]
#
#     if missing:
#         error(
#             "required_files",
#             "Arquivos obrigatórios ausentes.",
#             missing,
#         )
#         return False
#
#     ok(
#         "required_files",
#         "Todos os arquivos obrigatórios existem.",
#     )
#
#     return True
#
#
# # ============================================================
# # CHECK 2 — STATE MAPPING
# # ============================================================
#
# def check_state_mapping(
#     state_hourly: pd.DataFrame,
# ) -> dict:
#     with open(
#         STATE_MAPPING_PATH,
#         encoding="utf-8",
#     ) as f:
#         mapping = json.load(f)
#
#     mapping_df = pd.DataFrame(mapping)
#
#     required_cols = {
#         "id_estado",
#         "nom_estado",
#         "state_idx",
#     }
#
#     missing = (
#         required_cols
#         - set(mapping_df.columns)
#     )
#
#     if missing:
#         error(
#             "state_mapping_columns",
#             "Colunas ausentes no state_mapping.",
#             sorted(missing),
#         )
#         return {}
#
#     if mapping_df["state_idx"].duplicated().any():
#         error(
#             "state_mapping_unique_idx",
#             "Existem state_idx duplicados.",
#         )
#
#     if mapping_df["id_estado"].duplicated().any():
#         error(
#             "state_mapping_unique_state",
#             "Existem id_estado duplicados.",
#         )
#
#     mapping_states = set(
#         mapping_df["id_estado"]
#     )
#
#     data_states = set(
#         state_hourly["id_estado"]
#         .dropna()
#         .unique()
#     )
#
#     if mapping_states != data_states:
#         error(
#             "state_mapping_vs_state_hourly",
#             "Estados diferentes entre state_mapping e state_hourly.",
#             {
#                 "only_mapping": sorted(
#                     mapping_states - data_states
#                 ),
#                 "only_data": sorted(
#                     data_states - mapping_states
#                 ),
#             },
#         )
#     else:
#         ok(
#             "state_mapping_vs_state_hourly",
#             "Estados coincidem entre mapping e state_hourly.",
#             sorted(mapping_states),
#         )
#
#     missing_expected = (
#         EXPECTED_MODEL_STATES
#         - mapping_states
#     )
#
#     unexpected = (
#         mapping_states
#         - EXPECTED_MODEL_STATES
#     )
#
#     if missing_expected:
#         error(
#             "expected_model_states",
#             "Unidades esperadas do modelo ausentes.",
#             sorted(missing_expected),
#         )
#
#     if unexpected:
#         warning(
#             "unexpected_model_states",
#             "Existem unidades além das esperadas no modelo.",
#             sorted(unexpected),
#         )
#
#     else:
#         ok(
#             "expected_model_states",
#             "Unidades espaciais do modelo estão conforme esperado.",
#             sorted(mapping_states),
#         )
#
#     expected_idx = set(
#         range(len(mapping_df))
#     )
#
#     actual_idx = set(
#         mapping_df["state_idx"]
#         .astype(int)
#     )
#
#     if expected_idx != actual_idx:
#         error(
#             "state_idx_contiguous",
#             "state_idx não é sequencial de 0 até N-1.",
#             sorted(actual_idx),
#         )
#     else:
#         ok(
#             "state_idx_contiguous",
#             "state_idx é sequencial.",
#         )
#
#     return {
#         row["id_estado"]: int(
#             row["state_idx"]
#         )
#         for _, row in mapping_df.iterrows()
#     }
#
#
# # ============================================================
# # CHECK 3 — STATE HOURLY
# # ============================================================
#
# def check_state_hourly(
#     state_hourly: pd.DataFrame,
# ) -> None:
#
#     required = [
#         "id_estado",
#         "nom_estado",
#         "id_subsistema",
#         "nom_subsistema",
#         "datetime",
#         "GerRenEOL_MW",
#         "load_subsystem_MWmed",
#         "curtailment_MWmed",
#         "curtailment_MWh",
#         "curtailment_minutes",
#         "curtailment_flag",
#         "complete_eolic_hour",
#     ]
#
#     missing = [
#         c
#         for c in required
#         if c not in state_hourly.columns
#     ]
#
#     if missing:
#         error(
#             "state_hourly_columns",
#             "Colunas obrigatórias ausentes.",
#             missing,
#         )
#         return
#
#     if not pd.api.types.is_datetime64_any_dtype(
#         state_hourly["datetime"]
#     ):
#         error(
#             "state_hourly_datetime_type",
#             "datetime não está em formato datetime.",
#         )
#     else:
#         ok(
#             "state_hourly_datetime_type",
#             "datetime está em formato temporal.",
#         )
#
#     duplicated = state_hourly.duplicated(
#         [
#             "id_estado",
#             "datetime",
#         ]
#     ).sum()
#
#     if duplicated:
#         error(
#             "state_hourly_duplicates",
#             "Existem chaves estado × hora duplicadas.",
#             int(duplicated),
#         )
#     else:
#         ok(
#             "state_hourly_duplicates",
#             "Não existem duplicidades estado × hora.",
#         )
#
#     invalid_flag = (
#         ~state_hourly[
#             "curtailment_flag"
#         ].isin([0, 1])
#     ).sum()
#
#     if invalid_flag:
#         error(
#             "curtailment_flag_values",
#             "curtailment_flag possui valores diferentes de 0/1.",
#             int(invalid_flag),
#         )
#     else:
#         ok(
#             "curtailment_flag_values",
#             "curtailment_flag contém apenas 0/1.",
#         )
#
#     mismatch_mwh = (
#         (
#             state_hourly[
#                 "curtailment_MWh"
#             ]
#             - state_hourly[
#                 "curtailment_MWmed"
#             ]
#         )
#         .abs()
#         > 1e-5
#     ).sum()
#
#     if mismatch_mwh:
#         error(
#             "mwh_vs_mwmed",
#             "curtailment_MWh diverge de curtailment_MWmed.",
#             int(mismatch_mwh),
#         )
#     else:
#         ok(
#             "mwh_vs_mwmed",
#             "curtailment_MWh é consistente com MWmed em base horária.",
#         )
#
#     expected_flag = (
#         (
#             state_hourly[
#                 "curtailment_MWmed"
#             ].fillna(0)
#             > 0
#         )
#         |
#         (
#             state_hourly[
#                 "curtailment_minutes"
#             ].fillna(0)
#             > 0
#         )
#         |
#         (
#             state_hourly[
#                 EXPECTED_REASON_COLS
#             ]
#             .max(axis=1)
#             .fillna(0)
#             > 0
#         )
#     ).astype("int8")
#
#     flag_mismatch = (
#         expected_flag
#         != state_hourly[
#             "curtailment_flag"
#         ]
#     ).sum()
#
#     if flag_mismatch:
#         error(
#             "flag_target_consistency",
#             "curtailment_flag não é consistente com os alvos de curtailment.",
#             int(flag_mismatch),
#         )
#     else:
#         ok(
#             "flag_target_consistency",
#             "curtailment_flag é consistente com os alvos.",
#         )
#
#     numeric_cols = state_hourly.select_dtypes(
#         include=[np.number]
#     ).columns
#
#     nan_summary = (
#         state_hourly[
#             numeric_cols
#         ]
#         .isna()
#         .sum()
#     )
#
#     informative_nan = (
#         nan_summary[
#             nan_summary > 0
#         ]
#         .sort_values(
#             ascending=False
#         )
#     )
#
#     if not informative_nan.empty:
#         warning(
#             "state_hourly_missing_values",
#             "Existem valores ausentes em colunas numéricas; alguns podem ser warm-up de features.",
#             informative_nan.to_dict(),
#         )
#     else:
#         ok(
#             "state_hourly_missing_values",
#             "Nenhum NaN nas colunas numéricas.",
#         )
#
#     inf_count = np.isinf(
#         state_hourly[
#             numeric_cols
#         ].to_numpy()
#     ).sum()
#
#     if inf_count:
#         error(
#             "state_hourly_inf",
#             "Existem valores infinitos.",
#             int(inf_count),
#         )
#     else:
#         ok(
#             "state_hourly_inf",
#             "Nenhum valor infinito.",
#         )
#
#
# # ============================================================
# # CHECK 4 — GRADE TEMPORAL ESTADUAL
# # ============================================================
#
# def check_state_temporal_grid(
#     state_hourly: pd.DataFrame,
# ) -> None:
#
#     report = []
#
#     for state, group in state_hourly.groupby(
#         "id_estado",
#         observed=True,
#     ):
#         g = (
#             group[
#                 [
#                     "datetime",
#                     "complete_eolic_hour",
#                 ]
#             ]
#             .drop_duplicates("datetime")
#             .sort_values("datetime")
#         )
#
#         diffs = g["datetime"].diff().dropna()
#
#         wrong_frequency = (
#             diffs
#             != pd.Timedelta(hours=1)
#         ).sum()
#
#         missing_hours = int(
#             (
#                 g["datetime"]
#                 .max()
#                 - g["datetime"]
#                 .min()
#             )
#             / pd.Timedelta(hours=1)
#             + 1
#             - len(g)
#         )
#
#         complete_pct = (
#             g["complete_eolic_hour"]
#             .mean()
#             * 100
#         )
#
#         report.append(
#             {
#                 "id_estado": state,
#                 "min_datetime": g["datetime"].min(),
#                 "max_datetime": g["datetime"].max(),
#                 "n_hours": len(g),
#                 "n_wrong_intervals": int(
#                     wrong_frequency
#                 ),
#                 "n_missing_hours": missing_hours,
#                 "pct_complete_eolic_hour": (
#                     complete_pct
#                 ),
#             }
#         )
#
#         if wrong_frequency:
#             error(
#                 f"state_time_grid_{state}",
#                 "Existem intervalos diferentes de 1 hora.",
#                 int(wrong_frequency),
#             )
#
#         if missing_hours:
#             warning(
#                 f"state_missing_hours_{state}",
#                 "Existem horas ausentes na grade temporal.",
#                 missing_hours,
#             )
#
#     report_df = pd.DataFrame(report)
#
#     report_df.to_csv(
#         TRAIN_DIR / "temporal_state_report.csv",
#         index=False,
#         encoding="utf-8-sig",
#     )
#
#     ok(
#         "state_temporal_grid_report",
#         "Relatório temporal estadual salvo.",
#         str(
#             TRAIN_DIR
#             / "temporal_state_report.csv"
#         ),
#     )
#
#
# # ============================================================
# # CHECK 5 — RAW EOLIC
# # ============================================================
#
# def inspect_raw_eolic() -> None:
#     files = list_csvs(
#         EOLIC_DIR
#     )
#
#     rows = []
#
#     for path in files:
#         header = set(
#             read_csv_header(path)
#         )
#
#         if "din_instante" not in header:
#             error(
#                 f"raw_eolic_columns_{path.name}",
#                 "din_instante ausente.",
#             )
#             continue
#
#         timestamps = []
#
#         for chunk in pd.read_csv(
#             path,
#             usecols=["din_instante"],
#             chunksize=CHUNKSIZE,
#             **CSV_KW,
#         ):
#             timestamps.append(
#                 parse_datetime(
#                     chunk["din_instante"]
#                 )
#             )
#
#         dates = pd.concat(
#             timestamps,
#             ignore_index=True,
#         )
#
#         stats = date_range_stats(
#             dates,
#             "30min",
#         )
#
#         rows.append(
#             {
#                 "file": path.name,
#                 **stats,
#             }
#         )
#
#         if stats["n_gaps"] > 0:
#             warning(
#                 f"raw_eolic_gaps_{path.name}",
#                 "Existem gaps na sequência de timestamps do arquivo.",
#                 {
#                     "n_gaps": stats["n_gaps"],
#                     "max_gap_hours": stats[
#                         "max_gap_hours"
#                     ],
#                 },
#             )
#
#     pd.DataFrame(
#         rows
#     ).to_csv(
#         TRAIN_DIR / "raw_eolic_temporal_report.csv",
#         index=False,
#         encoding="utf-8-sig",
#     )
#
#     ok(
#         "raw_eolic_temporal",
#         "Relatório temporal dos arquivos eólicos salvo.",
#     )
#
#
# # ============================================================
# # CHECK 6 — RAW LOAD
# # ============================================================
#
# def inspect_raw_hourly_dataset(
#     directory: Path,
#     dataset_name: str,
# ) -> None:
#
#     files = list_csvs(
#         directory
#     )
#
#     rows = []
#
#     for path in files:
#         header = set(
#             read_csv_header(path)
#         )
#
#         if "din_instante" not in header:
#             error(
#                 f"{dataset_name}_columns_{path.name}",
#                 "din_instante ausente.",
#             )
#             continue
#
#         chunks = []
#
#         for chunk in pd.read_csv(
#             path,
#             usecols=["din_instante"],
#             chunksize=CHUNKSIZE,
#             **CSV_KW,
#         ):
#             chunks.append(
#                 parse_datetime(
#                     chunk["din_instante"]
#                 )
#             )
#
#         dates = pd.concat(
#             chunks,
#             ignore_index=True,
#         )
#
#         stats = date_range_stats(
#             dates,
#             "1h",
#         )
#
#         rows.append(
#             {
#                 "file": path.name,
#                 **stats,
#             }
#         )
#
#         if stats["n_gaps"] > 0:
#             warning(
#                 f"{dataset_name}_gaps_{path.name}",
#                 "Existem gaps no dataset horário.",
#                 {
#                     "n_gaps": stats["n_gaps"],
#                     "max_gap_hours": stats[
#                         "max_gap_hours"
#                     ],
#                 },
#             )
#
#     output = (
#         TRAIN_DIR
#         / f"raw_{dataset_name}_temporal_report.csv"
#     )
#
#     pd.DataFrame(
#         rows
#     ).to_csv(
#         output,
#         index=False,
#         encoding="utf-8-sig",
#     )
#
#     ok(
#         f"raw_{dataset_name}_temporal",
#         "Relatório temporal salvo.",
#         str(output),
#     )
#
#
# # ============================================================
# # CHECK 7 — INTERVALO GLOBAL DOS DATASETS
# # ============================================================
#
# def get_parquet_time_range(
#     path: Path,
# ) -> tuple[pd.Timestamp, pd.Timestamp]:
#     df = pd.read_parquet(
#         path,
#         columns=["datetime"],
#     )
#
#     return (
#         df["datetime"].min(),
#         df["datetime"].max(),
#     )
#
#
# def check_global_time_ranges() -> None:
#
#     state_min, state_max = (
#         get_parquet_time_range(
#             STATE_HOURLY_PATH
#         )
#     )
#
#     rows = [
#         {
#             "dataset": "state_hourly",
#             "min_datetime": state_min,
#             "max_datetime": state_max,
#         }
#     ]
#
#     logger.info(
#         "state_hourly: %s -> %s",
#         state_min,
#         state_max,
#     )
#
#     for directory, name in [
#         (
#             EOLIC_DIR,
#             "eolic_raw",
#         ),
#         (
#             LOAD_DIR,
#             "load_raw",
#         ),
#         (
#             INTERCHANGE_DIR,
#             "interchange_raw",
#         ),
#     ]:
#         files = list_csvs(
#             directory
#         )
#
#         all_min = None
#         all_max = None
#
#         for path in files:
#             for chunk in pd.read_csv(
#                 path,
#                 usecols=["din_instante"],
#                 chunksize=CHUNKSIZE,
#                 **CSV_KW,
#             ):
#                 dt = parse_datetime(
#                     chunk["din_instante"]
#                 ).dropna()
#
#                 if dt.empty:
#                     continue
#
#                 chunk_min = dt.min()
#                 chunk_max = dt.max()
#
#                 all_min = (
#                     chunk_min
#                     if all_min is None
#                     else min(
#                         all_min,
#                         chunk_min,
#                     )
#                 )
#
#                 all_max = (
#                     chunk_max
#                     if all_max is None
#                     else max(
#                         all_max,
#                         chunk_max,
#                     )
#                 )
#
#         rows.append(
#             {
#                 "dataset": name,
#                 "min_datetime": all_min,
#                 "max_datetime": all_max,
#             }
#         )
#
#     report = pd.DataFrame(
#         rows
#     )
#
#     report.to_csv(
#         TRAIN_DIR
#         / "global_time_ranges.csv",
#         index=False,
#         encoding="utf-8-sig",
#     )
#
#     logger.info(
#         "\n%s",
#         report.to_string(
#             index=False
#         ),
#     )
#
#     # Verifica compatibilidade dos limites.
#     source_ranges = report[
#         report["dataset"]
#         != "state_hourly"
#     ]
#
#     earlier = (
#         source_ranges[
#             "min_datetime"
#         ] > state_min
#     )
#
#     later = (
#         source_ranges[
#             "max_datetime"
#         ] < state_max
#     )
#
#     if earlier.any():
#         warning(
#             "global_start_alignment",
#             "Alguma fonte começa depois do state_hourly.",
#             source_ranges.loc[
#                 earlier
#             ].to_dict("records"),
#         )
#
#     if later.any():
#         warning(
#             "global_end_alignment",
#             "Alguma fonte termina antes do state_hourly.",
#             source_ranges.loc[
#                 later
#             ].to_dict("records"),
#         )
#
#     ok(
#         "global_time_ranges",
#         "Intervalos temporais globais comparados.",
#         str(
#             TRAIN_DIR
#             / "global_time_ranges.csv"
#         ),
#     )
#
#
# # ============================================================
# # CHECK 8 — EÓLICO: 2 MEIAS-HORAS POR HORA
# # ============================================================
#
# def check_eolic_halfhours(
#     state_hourly: pd.DataFrame,
# ) -> None:
#
#     if "n_halfhours" not in state_hourly:
#         error(
#             "eolic_halfhours_column",
#             "Coluna n_halfhours ausente.",
#         )
#         return
#
#     counts = (
#         state_hourly[
#             "n_halfhours"
#         ]
#         .value_counts(
#             dropna=False
#         )
#         .sort_index()
#     )
#
#     invalid = (
#         state_hourly[
#             "n_halfhours"
#         ]
#         != 2
#     ).sum()
#
#     if invalid:
#         warning(
#             "eolic_two_halfhours",
#             "Existem horas eólicas que não possuem exatamente 2 meias-horas.",
#             int(invalid),
#         )
#     else:
#         ok(
#             "eolic_two_halfhours",
#             "Todas as horas eólicas possuem 2 meias-horas.",
#         )
#
#     counts.to_csv(
#         TRAIN_DIR / "eolic_halfhour_distribution.csv",
#         header=["count"],
#     )
#
#
# # ============================================================
# # CHECK 9 — FEATURES
# # ============================================================
#
# def check_features(
#     state_hourly: pd.DataFrame,
# ) -> list[str]:
#
#     with open(
#         FEATURE_COLUMNS_PATH,
#         encoding="utf-8",
#     ) as f:
#         features = json.load(f)
#
#     if not isinstance(
#         features,
#         list,
#     ):
#         error(
#             "feature_columns_type",
#             "feature_columns.json não contém uma lista.",
#         )
#         return []
#
#     missing = [
#         col
#         for col in features
#         if col not in state_hourly.columns
#     ]
#
#     if missing:
#         error(
#             "feature_columns_vs_state_hourly",
#             "Existem features listadas no JSON que não estão no parquet.",
#             missing,
#         )
#     else:
#         ok(
#             "feature_columns_vs_state_hourly",
#             "Todas as features existem no state_hourly.",
#             len(features),
#         )
#
#     duplicates = pd.Series(
#         features
#     ).duplicated()
#
#     if duplicates.any():
#         error(
#             "feature_columns_duplicates",
#             "Existem features duplicadas.",
#         )
#     else:
#         ok(
#             "feature_columns_duplicates",
#             "Não existem features duplicadas.",
#         )
#
#     return features
#
#
# # ============================================================
# # CHECK 10 — SEQUÊNCIAS
# # ============================================================
#
# def check_sequences(
#     mapping_dict: dict,
#     feature_columns: list[str],
# ) -> None:
#
#     arrays = {
#         "X": np.load(
#             X_PATH,
#             mmap_mode="r",
#         ),
#         "y_flag": np.load(
#             Y_FLAG_PATH,
#             mmap_mode="r",
#         ),
#         "y_mw": np.load(
#             Y_MW_PATH,
#             mmap_mode="r",
#         ),
#         "y_mwh": np.load(
#             Y_MWH_PATH,
#             mmap_mode="r",
#         ),
#         "y_minutes": np.load(
#             Y_MINUTES_PATH,
#             mmap_mode="r",
#         ),
#         "y_reason": np.load(
#             Y_REASON_PATH,
#             mmap_mode="r",
#         ),
#         "state_idx": np.load(
#             STATE_IDX_PATH,
#             mmap_mode="r",
#         ),
#         "end_datetime": np.load(
#             END_DATETIME_PATH,
#             mmap_mode="r",
#         ),
#     }
#
#     n = arrays["X"].shape[0]
#
#     # --------------------------
#     # Shapes
#     # --------------------------
#
#     if arrays["X"].ndim != 3:
#         error(
#             "X_dimensions",
#             "X.npy deveria ser tridimensional.",
#             arrays["X"].shape,
#         )
#     else:
#         expected_shape = (
#             EXPECTED_SEQUENCE_LENGTH,
#             len(feature_columns),
#         )
#
#         if arrays["X"].shape[1:] != expected_shape:
#             error(
#                 "X_shape",
#                 "Shape de X não corresponde a 168 × n_features.",
#                 {
#                     "actual": arrays["X"].shape,
#                     "expected_tail": expected_shape,
#                 },
#             )
#         else:
#             ok(
#                 "X_shape",
#                 "Shape de X está correto.",
#                 arrays["X"].shape,
#             )
#
#     if arrays["y_reason"].ndim != 2:
#         error(
#             "y_reason_dimensions",
#             "y_reason deveria ser bidimensional.",
#             arrays["y_reason"].shape,
#         )
#
#     else:
#         if arrays["y_reason"].shape[1] != EXPECTED_REASON_COUNT:
#             error(
#                 "y_reason_shape",
#                 "y_reason deveria possuir 4 razões.",
#                 arrays["y_reason"].shape,
#             )
#         else:
#             ok(
#                 "y_reason_shape",
#                 "Shape de y_reason está correto.",
#                 arrays["y_reason"].shape,
#             )
#
#     # --------------------------
#     # Número de amostras
#     # --------------------------
#
#     for name, array in arrays.items():
#         if name == "X":
#             continue
#
#         if len(array) != n:
#             error(
#                 f"sequence_length_{name}",
#                 "Quantidade de amostras incompatível com X.",
#                 {
#                     "X": n,
#                     name: len(array),
#                 },
#             )
#
#     # --------------------------
#     # Dtypes
#     # --------------------------
#
#     ok(
#         "sequence_dtypes",
#         "Dtypes das sequências registrados.",
#         {
#             name: str(arr.dtype)
#             for name, arr in arrays.items()
#         },
#     )
#
#     # --------------------------
#     # State index
#     # --------------------------
#
#     valid_state_idx = set(
#         mapping_dict.values()
#     )
#
#     sequence_states = set(
#         np.unique(
#             arrays["state_idx"]
#         ).tolist()
#     )
#
#     if not sequence_states.issubset(
#         valid_state_idx
#     ):
#         error(
#             "sequence_state_idx",
#             "Existem state_idx nas sequências que não existem no mapping.",
#             sorted(
#                 sequence_states
#                 - valid_state_idx
#             ),
#         )
#     else:
#         ok(
#             "sequence_state_idx",
#             "state_idx das sequências está alinhado ao mapping.",
#             sorted(sequence_states),
#         )
#
#     # --------------------------
#     # End datetime
#     # --------------------------
#
#     end_dt = pd.to_datetime(
#         arrays["end_datetime"],
#         unit="ns",
#         errors="coerce",
#     )
#
#     if end_dt.isna().any():
#         error(
#             "end_datetime_valid",
#             "Existem end_datetime inválidos.",
#         )
#     else:
#         ok(
#             "end_datetime_valid",
#             "Todos os end_datetime são válidos.",
#         )
#
#     if not end_dt.is_monotonic_increasing:
#         warning(
#             "end_datetime_global_order",
#             "end_datetime não é globalmente crescente; isso pode ser normal quando as amostras estão agrupadas por estado.",
#         )
#
#     # --------------------------
#     # Flags
#     # --------------------------
#
#     flag_unique = set(
#         np.unique(
#             arrays["y_flag"]
#         ).tolist()
#     )
#
#     if not flag_unique.issubset(
#         {0, 1}
#     ):
#         error(
#             "y_flag_values",
#             "y_curtailment_flag possui valores diferentes de 0/1.",
#             sorted(flag_unique),
#         )
#     else:
#         ok(
#             "y_flag_values",
#             "y_curtailment_flag contém apenas 0/1.",
#         )
#
#     # --------------------------
#     # Reasons
#     # --------------------------
#
#     reason_values = np.unique(
#         arrays["y_reason"]
#     )
#
#     if not set(
#         reason_values.tolist()
#     ).issubset({0, 1}):
#         error(
#             "y_reason_values",
#             "y_reason possui valores diferentes de 0/1.",
#             reason_values.tolist(),
#         )
#     else:
#         ok(
#             "y_reason_values",
#             "y_reason contém apenas 0/1.",
#         )
#
#     # --------------------------
#     # NaN / Inf
#     # --------------------------
#
#     for name, array in arrays.items():
#
#         if not np.issubdtype(
#             array.dtype,
#             np.number,
#         ):
#             continue
#
#         # X pode ser muito grande; verificação em chunks.
#         if name == "X":
#             has_nan = False
#             has_inf = False
#
#             for start in range(
#                 0,
#                 n,
#                 10_000,
#             ):
#                 block = array[
#                     start:start + 10_000
#                 ]
#
#                 if np.isnan(
#                     block
#                 ).any():
#                     has_nan = True
#
#                 if np.isinf(
#                     block
#                 ).any():
#                     has_inf = True
#
#                 if has_nan or has_inf:
#                     break
#
#         else:
#             has_nan = np.isnan(
#                 array
#             ).any()
#
#             has_inf = np.isinf(
#                 array
#             ).any()
#
#         if has_nan:
#             error(
#                 f"sequence_nan_{name}",
#                 f"{name} contém NaN.",
#             )
#
#         if has_inf:
#             error(
#                 f"sequence_inf_{name}",
#                 f"{name} contém infinito.",
#             )
#
#     # --------------------------
#     # MW vs MWh
#     # --------------------------
#
#     mw = np.asarray(
#         arrays["y_mw"]
#     )
#
#     mwh = np.asarray(
#         arrays["y_mwh"]
#     )
#
#     mismatch = np.sum(
#         np.abs(mw - mwh)
#         > 1e-5
#     )
#
#     if mismatch:
#         error(
#             "sequence_mw_mwh",
#             "y_mw e y_mwh divergem.",
#             int(mismatch),
#         )
#     else:
#         ok(
#             "sequence_mw_mwh",
#             "y_mw e y_mwh são consistentes.",
#         )
#
#
# # ============================================================
# # CHECK 11 — COVERAGE REPORT
# # ============================================================
#
# def check_coverage_report(
#     state_hourly: pd.DataFrame,
# ) -> None:
#
#     if not COVERAGE_REPORT_PATH.exists():
#         warning(
#             "coverage_report",
#             "coverage_report.csv não encontrado.",
#         )
#         return
#
#     report = pd.read_csv(
#         COVERAGE_REPORT_PATH
#     )
#
#     expected = {
#         "id_estado",
#         "nom_estado",
#         "n_hours",
#         "n_complete_eolic_hours",
#         "pct_complete_eolic_hours",
#         "n_gerren_missing",
#         "n_load_missing",
#     }
#
#     missing = (
#         expected
#         - set(report.columns)
#     )
#
#     if missing:
#         error(
#             "coverage_report_columns",
#             "Colunas ausentes no coverage_report.",
#             sorted(missing),
#         )
#         return
#
#     coverage_states = set(
#         report["id_estado"]
#     )
#
#     actual_states = set(
#         state_hourly["id_estado"]
#     )
#
#     if coverage_states != actual_states:
#         error(
#             "coverage_report_states",
#             "Estados do coverage_report diferem do state_hourly.",
#             {
#                 "coverage_only": sorted(
#                     coverage_states
#                     - actual_states
#                 ),
#                 "data_only": sorted(
#                     actual_states
#                     - coverage_states
#                 ),
#             },
#         )
#     else:
#         ok(
#             "coverage_report_states",
#             "Estados do coverage_report coincidem.",
#         )
#
#
# # ============================================================
# # CHECK 12 — CONSISTÊNCIA DOS INTERVALOS DAS SEQUÊNCIAS
# # ============================================================
#
# def check_sequence_state_temporal_consistency() -> None:
#     """
#     Confere se, dentro de cada estado,
#     as amostras estão espaçadas por 1 hora
#     em end_datetime.
#
#     Não exige 1 hora globalmente porque
#     o dataset pode agrupar estados.
#     """
#     state_idx = np.load(
#         STATE_IDX_PATH,
#         mmap_mode="r",
#     )
#
#     end_datetime = pd.to_datetime(
#         np.load(
#             END_DATETIME_PATH,
#             mmap_mode="r",
#         ),
#         unit="ns",
#     )
#
#     rows = []
#
#     df = pd.DataFrame(
#         {
#             "state_idx": state_idx,
#             "end_datetime": end_datetime,
#         }
#     )
#
#     for state, group in df.groupby(
#         "state_idx"
#     ):
#         g = (
#             group
#             .sort_values("end_datetime")
#         )
#
#         diff = (
#             g["end_datetime"]
#             .diff()
#             .dropna()
#         )
#
#         wrong = (
#             diff
#             != pd.Timedelta(hours=1)
#         ).sum()
#
#         rows.append(
#             {
#                 "state_idx": state,
#                 "n_samples": len(g),
#                 "n_non_hourly_transitions": int(
#                     wrong
#                 ),
#                 "min_datetime": g[
#                     "end_datetime"
#                 ].min(),
#                 "max_datetime": g[
#                     "end_datetime"
#                 ].max(),
#             }
#         )
#
#         if wrong:
#             warning(
#                 f"sequence_time_{state}",
#                 "Existem saltos diferentes de 1 hora entre finais de sequência.",
#                 int(wrong),
#             )
#
#     pd.DataFrame(
#         rows
#     ).to_csv(
#         TRAIN_DIR
#         / "sequence_temporal_report.csv",
#         index=False,
#         encoding="utf-8-sig",
#     )
#
#
# # ============================================================
# # EXECUÇÃO
# # ============================================================
#
# def main() -> None:
#
#     RESULTS.clear()
#
#     logger.info(
#         "Iniciando Data Checking..."
#     )
#
#     if not check_required_files():
#         print_summary()
#         return
#
#     # ----------------------------------------
#     # Parquet principal
#     # ----------------------------------------
#
#     logger.info(
#         "Lendo state_hourly..."
#     )
#
#     state_hourly = pd.read_parquet(
#         STATE_HOURLY_PATH
#     )
#
#     ok(
#         "state_hourly_load",
#         "state_hourly carregado.",
#         {
#             "rows": len(state_hourly),
#             "columns": len(
#                 state_hourly.columns
#             ),
#         },
#     )
#
#     # ----------------------------------------
#     # Mapping
#     # ----------------------------------------
#
#     mapping_dict = check_state_mapping(
#         state_hourly
#     )
#
#     # ----------------------------------------
#     # State hourly
#     # ----------------------------------------
#
#     check_state_hourly(
#         state_hourly
#     )
#
#     check_state_temporal_grid(
#         state_hourly
#     )
#
#     check_eolic_halfhours(
#         state_hourly
#     )
#
#     # ----------------------------------------
#     # Features
#     # ----------------------------------------
#
#     feature_columns = check_features(
#         state_hourly
#     )
#
#     # ----------------------------------------
#     # Coverage
#     # ----------------------------------------
#
#     check_coverage_report(
#         state_hourly
#     )
#
#     # ----------------------------------------
#     # Raw datasets
#     # ----------------------------------------
#
#     inspect_raw_eolic()
#
#     inspect_raw_hourly_dataset(
#         LOAD_DIR,
#         "load",
#     )
#
#     inspect_raw_hourly_dataset(
#         INTERCHANGE_DIR,
#         "interchange",
#     )
#
#     # ----------------------------------------
#     # Global time ranges
#     # ----------------------------------------
#
#     check_global_time_ranges()
#
#     # ----------------------------------------
#     # Sequence files
#     # ----------------------------------------
#
#     check_sequences(
#         mapping_dict,
#         feature_columns,
#     )
#
#     check_sequence_state_temporal_consistency()
#
#     # ----------------------------------------
#     # Summary
#     # ----------------------------------------
#
#     summary = pd.DataFrame(
#         [
#             asdict(result)
#             for result in RESULTS
#         ]
#     )
#
#     summary_path = (
#         TRAIN_DIR
#         / "datacheck_results.csv"
#     )
#
#     summary.to_csv(
#         summary_path,
#         index=False,
#         encoding="utf-8-sig",
#     )
#
#     summary_json_path = (
#         TRAIN_DIR
#         / "datacheck_results.json"
#     )
#
#     with open(
#         summary_json_path,
#         "w",
#         encoding="utf-8",
#     ) as f:
#         json.dump(
#             [
#                 asdict(result)
#                 for result in RESULTS
#             ],
#             f,
#             ensure_ascii=False,
#             indent=2,
#             default=str,
#         )
#
#     print_summary()
#
#     errors = sum(
#         result.status == "ERROR"
#         for result in RESULTS
#     )
#
#     if errors:
#         raise SystemExit(
#             f"\nData checking terminou com {errors} erro(s)."
#         )
#
#     print(
#         "\nData checking concluído sem erros."
#     )
#
#
# if __name__ == "__main__":
#     main()

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DEFAULT_PROJECT_DIR = Path(".")
DEFAULT_OUTPUT_DIR = Path("data/training/data_audit")

CSV_KW = {
    "sep": ";",
    "encoding": "utf-8-sig",
}

CHUNKSIZE = 500_000
EXPECTED_SEQUENCE_LENGTH = 168

# Unidades espaciais deliberadas do modelo.
EXPECTED_MODEL_STATES = {
    "BA_SE",
    "CE",
    "PB",
    "PE",
    "PI",
    "RN",
}

REASON_COLS = [
    "reason_REL",
    "reason_CNF",
    "reason_ENE",
    "reason_PAR",
]

SEQUENCE_FILES = {
    "X": "X.npy",
    "y_flag": "y_curtailment_flag.npy",
    "y_mw": "y_curtailment_mwmed.npy",
    "y_mwh": "y_curtailment_mwh.npy",
    "y_minutes": "y_curtailment_minutes.npy",
    "y_reason": "y_reason.npy",
    "state_idx": "state_idx.npy",
    "end_datetime": "end_datetime.npy",
}

FEATURE_REGEX = {
    "lag": re.compile(r"^GerRenEOL_lag_(\d+)h$"),
    "roll": re.compile(r"^GerRenEOL_roll_(?:mean|std|max)_(\d+)h$"),
    "change": re.compile(r"^GerRenEOL_change_(\d+)h$"),
    "pct_change": re.compile(r"^GerRenEOL_pct_change_(\d+)h$"),
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("audit")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class AuditFinding:
    category: str
    severity: str
    status: str
    entity: str
    message: str
    evidence: dict[str, Any]


FINDINGS: list[AuditFinding] = []


def finding(
    category: str,
    severity: str,
    status: str,
    entity: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    FINDINGS.append(
        AuditFinding(
            category=category,
            severity=severity,
            status=status,
            entity=entity,
            message=message,
            evidence=evidence or {},
        )
    )


# ============================================================
# PATHS
# ============================================================

def build_paths(project_dir: Path) -> dict[str, Path]:
    raw = project_dir / "data" / "raw"
    processed = project_dir / "data" / "processed"
    training = project_dir / "data" / "training"
    sequence = training / "lstm_cnn"

    return {
        "state_hourly": processed / "state_hourly.parquet",
        "feature_columns": training / "feature_columns.json",
        "state_mapping": training / "state_mapping.json",
        "coverage_report": training / "coverage_report.csv",
        "eolic_raw": raw / "constrained_off_eolico",
        "load_raw": raw / "curva_carga",
        "interchange_raw": raw / "intercambio_subsistemas",
        "sequence_dir": sequence,
    }


# ============================================================
# GENERIC HELPERS
# ============================================================

def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def read_header(path: Path) -> list[str]:
    return pd.read_csv(
        path,
        nrows=0,
        **CSV_KW,
    ).columns.tolist()


def list_csvs(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.csv"))


def parse_datetime(series: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(
            series,
            errors="coerce",
        )
        .dt
        .tz_localize(None)
    )


def to_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


# ============================================================
# LOAD MAIN ARTIFACTS
# ============================================================

def load_main_artifacts(
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    logger.info("Lendo state_hourly...")
    state_hourly = pd.read_parquet(
        paths["state_hourly"]
    )

    logger.info("Lendo feature_columns.json...")
    with open(
        paths["feature_columns"],
        encoding="utf-8",
    ) as file:
        feature_columns = json.load(file)

    logger.info("Lendo state_mapping.json...")
    with open(
        paths["state_mapping"],
        encoding="utf-8",
    ) as file:
        mapping = pd.DataFrame(
            json.load(file)
        )

    return (
        state_hourly,
        feature_columns,
        mapping,
    )


# ============================================================
# 1. FEATURE MISSINGNESS AUDIT
# ============================================================

def feature_lookback(feature: str) -> int:
    """
    Retorna o warm-up esperado em linhas para uma feature.

    Exemplos:
      lag_168h       -> 168
      roll_mean_24h  -> 24
      change_24h     -> 24
      pct_change_24h -> 24
      hour_sin       -> 0
    """
    for regex in FEATURE_REGEX.values():
        match = regex.match(feature)
        if match:
            return int(match.group(1))

    return 0


def feature_type(feature: str) -> str:
    for name, regex in FEATURE_REGEX.items():
        if regex.match(feature):
            return name

    return "base"


def pct_change_zero_lag_column(feature: str) -> str | None:
    match = FEATURE_REGEX["pct_change"].match(feature)

    if not match:
        return None

    hours = match.group(1)
    return f"GerRenEOL_lag_{hours}h"


def classify_feature_nan(
    group: pd.DataFrame,
    feature: str,
) -> dict[str, int]:
    """
    Separa NaN em:
      - warm-up esperado;
      - denominador zero em pct_change;
      - inesperado.
    """
    missing_mask = group[feature].isna()

    total_missing = int(
        missing_mask.sum()
    )

    if total_missing == 0:
        return {
            "nan_count": 0,
            "warmup_nan": 0,
            "zero_denominator_nan": 0,
            "unexpected_nan": 0,
        }

    lookback = feature_lookback(feature)

    warmup_mask = (
        pd.Series(
            False,
            index=group.index,
        )
    )

    if lookback > 0:
        warmup_indices = group.index[:lookback]
        warmup_mask.loc[warmup_indices] = True

    warmup_nan = (
        missing_mask
        & warmup_mask
    )

    zero_denominator_mask = pd.Series(
        False,
        index=group.index,
    )

    lag_col = pct_change_zero_lag_column(
        feature
    )

    if lag_col and lag_col in group.columns:
        zero_denominator_mask = (
            missing_mask
            & group[lag_col].eq(0)
        )

    explained_mask = (
        warmup_nan
        | zero_denominator_mask
    )

    unexpected_nan = (
        missing_mask
        & ~explained_mask
    )

    return {
        "nan_count": total_missing,
        "warmup_nan": int(warmup_nan.sum()),
        "zero_denominator_nan": int(
            zero_denominator_mask.sum()
        ),
        "unexpected_nan": int(
            unexpected_nan.sum()
        ),
    }


def audit_feature_missingness(
    state_hourly: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for state, group in state_hourly.groupby(
        "id_estado",
        observed=True,
        sort=False,
    ):
        group = (
            group
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        for feature in feature_columns:
            result = classify_feature_nan(
                group,
                feature,
            )

            if result["nan_count"] == 0:
                continue

            lookback = feature_lookback(
                feature
            )

            if result["unexpected_nan"] > 0:
                status = "REVIEW"
                severity = "WARNING"
                message = (
                    "Há NaN além do warm-up esperado "
                    "e/ou além de denominador zero."
                )
            elif result["zero_denominator_nan"] > 0:
                status = "EXPLAINED_ZERO_DENOMINATOR"
                severity = "INFO"
                message = (
                    "NaN explicado por denominador zero "
                    "no percentual de variação."
                )
            else:
                status = "EXPECTED_WARMUP"
                severity = "INFO"
                message = (
                    "NaN explicado pelo warm-up da feature."
                )

            rows.append(
                {
                    "id_estado": state,
                    "feature": feature,
                    "feature_type": feature_type(feature),
                    "lookback_hours": lookback,
                    **result,
                    "status": status,
                    "severity": severity,
                    "message": message,
                }
            )

            finding(
                "feature_missingness",
                severity,
                status,
                f"{state}:{feature}",
                message,
                {
                    **result,
                    "lookback_hours": lookback,
                },
            )

    return rows


# ============================================================
# 2. RAW INTERCHANGE GAP AUDIT
# ============================================================

def audit_interchange_gaps(
    directory: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    files = list_csvs(directory)

    for path in files:
        header = set(
            read_header(path)
        )

        if "din_instante" not in header:
            finding(
                "interchange_gap",
                "ERROR",
                "INVALID",
                path.name,
                "din_instante não existe no arquivo.",
            )
            continue

        timestamp_parts = []

        for chunk in pd.read_csv(
            path,
            usecols=["din_instante"],
            chunksize=CHUNKSIZE,
            **CSV_KW,
        ):
            timestamp_parts.append(
                parse_datetime(
                    chunk["din_instante"]
                )
            )

        timestamps = (
            pd.concat(
                timestamp_parts,
                ignore_index=True,
            )
            .dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        if timestamps.empty:
            finding(
                "interchange_gap",
                "ERROR",
                "INVALID",
                path.name,
                "Nenhum timestamp válido encontrado.",
            )
            continue

        expected = pd.date_range(
            timestamps.min(),
            timestamps.max(),
            freq="h",
        )

        missing = expected.difference(
            pd.DatetimeIndex(timestamps)
        )

        diff = (
            timestamps.diff()
            .dropna()
        )

        gaps = diff[
            diff > pd.Timedelta(hours=1)
        ]

        duplicate_rows = 0

        # Verifica duplicidade temporal somente da estrutura temporal.
        # Múltiplos registros por hora são esperados no dataset.
        # Portanto, não classificamos isso como erro.
        duplicate_timestamp_count = int(
            len(timestamp_parts)
        )

        if missing.empty:
            finding(
                "interchange_gap",
                "INFO",
                "OK",
                path.name,
                "Nenhum timestamp horário ausente.",
                {
                    "min_datetime": to_timestamp(timestamps.min()),
                    "max_datetime": to_timestamp(timestamps.max()),
                    "missing_hours": 0,
                },
            )
        else:
            for missing_dt in missing:
                rows.append(
                    {
                        "file": path.name,
                        "gap_type": "MISSING_TIMESTAMP",
                        "previous_datetime": None,
                        "next_datetime": None,
                        "gap_hours": None,
                        "missing_datetime": to_timestamp(
                            missing_dt
                        ),
                    }
                )

            for gap_idx in gaps.index:
                previous_dt = timestamps.iloc[
                    gap_idx - 1
                ]
                next_dt = timestamps.iloc[
                    gap_idx
                ]

                gap = (
                    next_dt
                    - previous_dt
                )

                missing_count = int(
                    gap / pd.Timedelta(hours=1)
                ) - 1

                finding(
                    "interchange_gap",
                    "WARNING",
                    "REVIEW",
                    path.name,
                    "Gap temporal real no dataset bruto de intercâmbio.",
                    {
                        "previous_datetime": to_timestamp(
                            previous_dt
                        ),
                        "next_datetime": to_timestamp(
                            next_dt
                        ),
                        "gap_hours": (
                            gap.total_seconds()
                            / 3600
                        ),
                        "missing_hours": missing_count,
                    },
                )

                rows.append(
                    {
                        "file": path.name,
                        "gap_type": "CONTIGUITY_GAP",
                        "previous_datetime": to_timestamp(
                            previous_dt
                        ),
                        "next_datetime": to_timestamp(
                            next_dt
                        ),
                        "gap_hours": (
                            gap.total_seconds()
                            / 3600
                        ),
                        "missing_datetime": None,
                    }
                )

        if duplicate_rows:
            finding(
                "interchange_duplicate_timestamp",
                "WARNING",
                "REVIEW",
                path.name,
                "Duplicidades temporais detectadas.",
                {
                    "duplicate_rows": duplicate_rows
                },
            )

    return rows


# ============================================================
# 3. STATE HOURLY REAL TIME GAP AUDIT
# ============================================================

def audit_state_hourly_gaps(
    state_hourly: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []

    for state, group in state_hourly.groupby(
        "id_estado",
        observed=True,
        sort=False,
    ):
        dt = (
            group["datetime"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        diff = dt.diff()
        gap_indices = diff[
            diff > pd.Timedelta(hours=1)
        ].index

        if len(gap_indices) == 0:
            finding(
                "state_hourly_gap",
                "INFO",
                "OK",
                str(state),
                "Não existem gaps reais na grade horária estadual.",
            )
            continue

        for idx in gap_indices:
            previous_dt = dt.iloc[idx - 1]
            next_dt = dt.iloc[idx]
            delta = next_dt - previous_dt

            missing_hours = int(
                delta / pd.Timedelta(hours=1)
            ) - 1

            rows.append(
                {
                    "id_estado": state,
                    "previous_datetime": to_timestamp(
                        previous_dt
                    ),
                    "next_datetime": to_timestamp(
                        next_dt
                    ),
                    "gap_hours": (
                        delta.total_seconds()
                        / 3600
                    ),
                    "missing_hours": missing_hours,
                }
            )

            finding(
                "state_hourly_gap",
                "WARNING",
                "REVIEW",
                str(state),
                "Gap temporal real no state_hourly.",
                {
                    "previous_datetime": to_timestamp(
                        previous_dt
                    ),
                    "next_datetime": to_timestamp(
                        next_dt
                    ),
                    "missing_hours": missing_hours,
                },
            )

    return rows


# ============================================================
# 4. SEQUENCE BREAK AUDIT
# ============================================================

def list_invalid_features(
    group: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    invalid = (
        group[feature_columns]
        .isna()
        .any(axis=1)
    )

    invalid_rows = group.loc[
        invalid,
        feature_columns,
    ]

    columns = (
        invalid_rows
        .columns[
            invalid_rows
            .notna()
            .sum()
            < len(invalid_rows)
        ]
        .tolist()
    )

    return columns


def classify_sequence_break(
    state_group: pd.DataFrame,
    feature_columns: list[str],
    previous_end: pd.Timestamp,
    next_end: pd.Timestamp,
) -> dict[str, Any]:
    interval = state_group[
        (state_group["datetime"] > previous_end)
        & (state_group["datetime"] <= next_end)
    ].copy()

    interval = interval.sort_values(
        "datetime"
    )

    expected_times = pd.date_range(
        previous_end + pd.Timedelta(hours=1),
        next_end,
        freq="h",
    )

    actual_times = pd.DatetimeIndex(
        interval["datetime"]
    )

    missing_times = expected_times.difference(
        actual_times
    )

    feature_invalid = (
        interval[
            feature_columns
        ]
        .isna()
        .any(axis=1)
    )

    invalid_rows = interval.loc[
        feature_invalid
    ]

    invalid_features: set[str] = set()

    for _, row in invalid_rows.iterrows():
        invalid_features.update(
            [
                col
                for col in feature_columns
                if pd.isna(row[col])
            ]
        )

    zero_denominator_features: set[str] = set()

    for feature in invalid_features:
        lag_column = (
            pct_change_zero_lag_column(
                feature
            )
        )

        if (
            lag_column
            and lag_column in interval.columns
            and interval[
                lag_column
            ].eq(0).any()
        ):
            zero_denominator_features.add(
                feature
            )

    if len(missing_times):
        classification = "RAW_STATE_HOURLY_GAP"
        severity = "WARNING"
        message = (
            "O salto das sequências é explicado por "
            "horas ausentes no state_hourly."
        )
    elif len(invalid_rows):
        classification = (
            "FEATURE_INVALIDITY"
        )
        severity = "INFO"
        message = (
            "O salto é causado por linha(s) "
            "com feature inválida; não existe gap no relógio."
        )
    else:
        classification = "UNEXPLAINED"
        severity = "WARNING"
        message = (
            "O salto das sequências não foi explicado "
            "por gap temporal ou NaN de feature."
        )

    return {
        "previous_end_datetime": to_timestamp(
            previous_end
        ),
        "next_end_datetime": to_timestamp(
            next_end
        ),
        "endpoint_gap_hours": (
            (next_end - previous_end)
            .total_seconds()
            / 3600
        ),
        "missing_state_hourly_hours": len(
            missing_times
        ),
        "feature_invalid_rows": len(
            invalid_rows
        ),
        "invalid_features": sorted(
            invalid_features
        ),
        "zero_denominator_features": sorted(
            zero_denominator_features
        ),
        "classification": classification,
        "severity": severity,
        "message": message,
    }


def audit_sequence_breaks(
    state_hourly: pd.DataFrame,
    mapping: pd.DataFrame,
    feature_columns: list[str],
    sequence_dir: Path,
) -> list[dict[str, Any]]:
    state_idx = np.load(
        sequence_dir / SEQUENCE_FILES["state_idx"],
        mmap_mode="r",
    )

    end_datetime = pd.to_datetime(
        np.load(
            sequence_dir / SEQUENCE_FILES["end_datetime"],
            mmap_mode="r",
        ),
        unit="ns",
    )

    mapping_lookup = dict(
        zip(
            mapping["state_idx"].astype(int),
            mapping["id_estado"],
        )
    )

    sequence_df = pd.DataFrame(
        {
            "state_idx": state_idx,
            "end_datetime": end_datetime,
        }
    )

    rows: list[dict[str, Any]] = []

    for state_idx, seq_group in sequence_df.groupby(
        "state_idx",
        sort=False,
    ):
        state = mapping_lookup.get(
            int(state_idx),
            f"UNKNOWN_{state_idx}",
        )

        seq_group = (
            seq_group
            .sort_values("end_datetime")
            .reset_index(drop=True)
        )

        diffs = (
            seq_group["end_datetime"]
            .diff()
        )

        break_indices = diffs[
            diffs > pd.Timedelta(hours=1)
        ].index

        state_hourly_group = (
            state_hourly[
                state_hourly["id_estado"] == state
            ]
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        if not len(break_indices):
            finding(
                "sequence_break",
                "INFO",
                "OK",
                str(state),
                "Não existem saltos > 1h entre endpoints de sequência.",
            )
            continue

        for idx in break_indices:
            previous_end = seq_group.loc[
                idx - 1,
                "end_datetime",
            ]

            next_end = seq_group.loc[
                idx,
                "end_datetime",
            ]

            details = classify_sequence_break(
                state_hourly_group,
                feature_columns,
                previous_end,
                next_end,
            )

            details["state_idx"] = int(
                state_idx
            )
            details["id_estado"] = state

            rows.append(details)

            finding(
                "sequence_break",
                details["severity"],
                details["classification"],
                str(state),
                details["message"],
                details,
            )

    return rows


# ============================================================
# 5. SEQUENCE SHAPE / ALIGNMENT
# ============================================================

def audit_sequence_shapes(
    feature_columns: list[str],
    sequence_dir: Path,
) -> None:
    arrays = {
        name: np.load(
            sequence_dir / filename,
            mmap_mode="r",
        )
        for name, filename in SEQUENCE_FILES.items()
    }

    X = arrays["X"]
    n_samples = X.shape[0]

    if X.ndim != 3:
        finding(
            "sequence_shape",
            "ERROR",
            "INVALID",
            "X",
            "X não é tridimensional.",
            {"shape": list(X.shape)},
        )
    elif X.shape[1:] != (
        EXPECTED_SEQUENCE_LENGTH,
        len(feature_columns),
    ):
        finding(
            "sequence_shape",
            "ERROR",
            "INVALID",
            "X",
            "Shape de X não corresponde a 168 × n_features.",
            {
                "actual": list(X.shape),
                "expected": [
                    EXPECTED_SEQUENCE_LENGTH,
                    len(feature_columns),
                ],
            },
        )
    else:
        finding(
            "sequence_shape",
            "INFO",
            "OK",
            "X",
            "Shape de X está correto.",
            {"shape": list(X.shape)},
        )

    for name, array in arrays.items():
        if name == "X":
            continue

        if len(array) != n_samples:
            finding(
                "sequence_alignment",
                "ERROR",
                "INVALID",
                name,
                "Número de amostras diferente de X.",
                {
                    "X": n_samples,
                    name: len(array),
                },
            )

    flag_values = set(
        np.unique(
            arrays["y_flag"]
        ).tolist()
    )

    if not flag_values.issubset({0, 1}):
        finding(
            "sequence_target",
            "ERROR",
            "INVALID",
            "y_flag",
            "Valores diferentes de 0/1.",
            {"values": sorted(flag_values)},
        )

    reason_values = set(
        np.unique(
            arrays["y_reason"]
        ).tolist()
    )

    if not reason_values.issubset({0, 1}):
        finding(
            "sequence_target",
            "ERROR",
            "INVALID",
            "y_reason",
            "Valores diferentes de 0/1.",
            {"values": sorted(reason_values)},
        )

    if arrays["y_reason"].ndim != 2 or arrays["y_reason"].shape[1] != 4:
        finding(
            "sequence_target",
            "ERROR",
            "INVALID",
            "y_reason",
            "y_reason não possui shape [amostras, 4].",
            {"shape": list(arrays["y_reason"].shape)},
        )

    mw = np.asarray(
        arrays["y_mw"]
    )
    mwh = np.asarray(
        arrays["y_mwh"]
    )

    mismatch = int(
        np.sum(
            np.abs(mw - mwh) > 1e-5
        )
    )

    if mismatch:
        finding(
            "sequence_target",
            "ERROR",
            "INVALID",
            "y_mw_vs_y_mwh",
            "MW e MWh diferem.",
            {"mismatch_count": mismatch},
        )


# ============================================================
# 6. GLOBAL TIME RANGE AUDIT
# ============================================================

def collect_raw_range(
    directory: Path,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    all_min = None
    all_max = None

    for path in list_csvs(directory):
        header = set(
            read_header(path)
        )

        if "din_instante" not in header:
            continue

        for chunk in pd.read_csv(
            path,
            usecols=["din_instante"],
            chunksize=CHUNKSIZE,
            **CSV_KW,
        ):
            dt = parse_datetime(
                chunk["din_instante"]
            ).dropna()

            if dt.empty:
                continue

            chunk_min = dt.min()
            chunk_max = dt.max()

            all_min = (
                chunk_min
                if all_min is None
                else min(all_min, chunk_min)
            )

            all_max = (
                chunk_max
                if all_max is None
                else max(all_max, chunk_max)
            )

    return all_min, all_max


def audit_global_time_ranges(
    state_hourly: pd.DataFrame,
    paths: dict[str, Path],
) -> list[dict[str, Any]]:
    rows = []

    state_min = state_hourly["datetime"].min()
    state_max = state_hourly["datetime"].max()

    rows.append(
        {
            "dataset": "state_hourly",
            "min_datetime": to_timestamp(state_min),
            "max_datetime": to_timestamp(state_max),
        }
    )

    for directory, name in [
        (paths["eolic_raw"], "eolic_raw"),
        (paths["load_raw"], "load_raw"),
        (paths["interchange_raw"], "interchange_raw"),
    ]:
        min_dt, max_dt = collect_raw_range(
            directory
        )

        rows.append(
            {
                "dataset": name,
                "min_datetime": to_timestamp(min_dt),
                "max_datetime": to_timestamp(max_dt),
            }
        )

        if min_dt is not None and min_dt > state_min:
            finding(
                "global_time_range",
                "WARNING",
                "REVIEW",
                name,
                "A fonte começa depois do state_hourly.",
                {
                    "source_start": to_timestamp(min_dt),
                    "state_start": to_timestamp(state_min),
                },
            )

        if max_dt is not None and max_dt < state_max:
            finding(
                "global_time_range",
                "WARNING",
                "REVIEW",
                name,
                "A fonte termina antes do state_hourly.",
                {
                    "source_end": to_timestamp(max_dt),
                    "state_end": to_timestamp(state_max),
                },
            )

    return rows


# ============================================================
# 7. HUMAN-READABLE REPORT
# ============================================================

def build_markdown_report(
    output_dir: Path,
    findings: list[AuditFinding],
) -> None:
    lines = [
        "# Data Audit Report",
        "",
        "Relatório auditável da preparação do dataset estadual.",
        "",
        "## Resumo",
        "",
    ]

    counts = (
        pd.DataFrame(
            [asdict(item) for item in findings]
        )["status"]
        .value_counts()
        .to_dict()
    ) if findings else {}

    lines.extend(
        [
            f"- OK: {counts.get('OK', 0)}",
            f"- INFO: {counts.get('INFO', 0)}",
            f"- REVIEW: {counts.get('REVIEW', 0)}",
            f"- INVALID: {counts.get('INVALID', 0)}",
            "",
            "## Findings",
            "",
            "| Categoria | Severidade | Status | Entidade | Mensagem |",
            "|---|---|---|---|---|",
        ]
    )

    for item in findings:
        message = (
            item.message
            .replace("|", "\\|")
            .replace("\n", " ")
        )

        lines.append(
            f"| {item.category} | {item.severity} | "
            f"{item.status} | {item.entity} | {message} |"
        )

    lines.extend(
        [
            "",
            "## Arquivos de evidência",
            "",
            "- `feature_missingness_audit.csv`",
            "- `interchange_gap_audit.csv`",
            "- `state_hourly_gap_audit.csv`",
            "- `sequence_break_audit.csv`",
            "- `global_time_range_audit.csv`",
            "- `audit_findings.csv`",
            "- `audit_findings.json`",
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_dir / "data_audit_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline auditável para investigar "
            "warnings do dataset estadual."
        )
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Diretório raiz do projeto.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diretório de saída dos relatórios.",
    )

    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else project_dir / args.output_dir
    )

    paths = build_paths(
        project_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required = [
        paths["state_hourly"],
        paths["feature_columns"],
        paths["state_mapping"],
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Arquivos principais ausentes:\n"
            + "\n".join(missing)
        )

    logger.info(
        "Iniciando auditoria em %s",
        project_dir,
    )

    state_hourly, feature_columns, mapping = (
        load_main_artifacts(paths)
    )

    # --------------------------------------------------------
    # A. Missingness das features
    # --------------------------------------------------------

    logger.info(
        "Auditando NaN das features..."
    )

    feature_rows = audit_feature_missingness(
        state_hourly,
        feature_columns,
    )

    save_csv(
        feature_rows,
        output_dir / "feature_missingness_audit.csv",
    )

    # --------------------------------------------------------
    # B. Gaps reais no state_hourly
    # --------------------------------------------------------

    logger.info(
        "Auditando gaps reais do state_hourly..."
    )

    state_gap_rows = audit_state_hourly_gaps(
        state_hourly
    )

    save_csv(
        state_gap_rows,
        output_dir / "state_hourly_gap_audit.csv",
    )

    # --------------------------------------------------------
    # C. Gaps reais do intercâmbio bruto
    # --------------------------------------------------------

    logger.info(
        "Auditando gaps do intercâmbio bruto..."
    )

    interchange_gap_rows = audit_interchange_gaps(
        paths["interchange_raw"]
    )

    save_csv(
        interchange_gap_rows,
        output_dir / "interchange_gap_audit.csv",
    )

    # --------------------------------------------------------
    # D. Gaps / breaks das sequências
    # --------------------------------------------------------

    sequence_files_exist = all(
        (
            paths["sequence_dir"]
            / filename
        ).exists()
        for filename in SEQUENCE_FILES.values()
    )

    if sequence_files_exist:
        logger.info(
            "Auditando sequências..."
        )

        audit_sequence_shapes(
            feature_columns,
            paths["sequence_dir"],
        )

        sequence_break_rows = (
            audit_sequence_breaks(
                state_hourly,
                mapping,
                feature_columns,
                paths["sequence_dir"],
            )
        )

        save_csv(
            sequence_break_rows,
            output_dir / "sequence_break_audit.csv",
        )
    else:
        finding(
            "sequence_files",
            "WARNING",
            "REVIEW",
            "lstm_cnn",
            "Nem todos os arquivos de sequência existem.",
        )

    # --------------------------------------------------------
    # E. Intervalos globais
    # --------------------------------------------------------

    logger.info(
        "Auditando intervalos globais..."
    )

    global_rows = audit_global_time_ranges(
        state_hourly,
        paths,
    )

    save_csv(
        global_rows,
        output_dir / "global_time_range_audit.csv",
    )

    # --------------------------------------------------------
    # F. Consolidação
    # --------------------------------------------------------

    findings_dict = [
        asdict(item)
        for item in FINDINGS
    ]

    findings_flat = []

    for item in FINDINGS:
        row = {
            "category": item.category,
            "severity": item.severity,
            "status": item.status,
            "entity": item.entity,
            "message": item.message,
        }

        for key, value in item.evidence.items():
            row[f"evidence_{key}"] = value

        findings_flat.append(row)

    save_csv(
        findings_flat,
        output_dir / "audit_findings.csv",
    )

    save_json(
        findings_dict,
        output_dir / "audit_findings.json",
    )

    build_markdown_report(
        output_dir,
        FINDINGS,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    summary = pd.Series(
        [
            item.status
            for item in FINDINGS
        ]
    ).value_counts()

    print("\n" + "=" * 100)
    print("AUDITORIA CONCLUÍDA")
    print("=" * 100)

    for status in [
        "OK",
        "INFO",
        "REVIEW",
        "INVALID",
    ]:
        print(
            f"{status:10}: "
            f"{int(summary.get(status, 0))}"
        )

    print("\nArquivos:")
    for path in sorted(
        output_dir.iterdir()
    ):
        print(
            f"  {path}"
        )

    print("\nClassificações principais:")
    print(
        "  EXPECTED_WARMUP              = NaN esperado pelo warm-up"
    )
    print(
        "  EXPLAINED_ZERO_DENOMINATOR   = NaN causado por lag = 0"
    )
    print(
        "  FEATURE_INVALIDITY           = sequência interrompida por feature inválida"
    )
    print(
        "  RAW_STATE_HOURLY_GAP         = gap real no state_hourly"
    )
    print(
        "  UNEXPLAINED                  = quebra sem explicação encontrada"
    )

    invalid = int(
        summary.get("INVALID", 0)
    )

    if invalid:
        raise SystemExit(
            f"\nAuditoria terminou com {invalid} achado(s) INVALID."
        )


if __name__ == "__main__":
    main()
