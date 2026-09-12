\## Phase 1 — Data Ingestion



\*\*Objective:\*\* Parse the raw e-nose breath sensor files into a validated, structured format with subject/group linkage, ready for feature engineering.



\### Data Source

Electronic nose dataset for COPD detection from smokers and healthy people through exhaled breath analysis (Durán Acevedo et al., 2021, \*Data in Brief\*; Mendeley Data DOI 10.17632/h5pcn99zw4).



\### Raw Files

| File | Rows (timepoints) | Columns | Parsed as |

|---|---|---|---|

| `COPD.txt` | 4000 | 320 | 40 samples = 20 subjects × 2 measurements |

| `CONTROL.txt` | 4000 | 160 | 20 samples = 10 subjects × 2 measurements |

| `SMOKERS.txt` | 4000 | 128 | 16 samples (subject linkage unresolved — see below) |

| `AIR.txt` | 4080 | 80 | 10 samples (ambient baseline, no subject) |

| `General\_data\_from\_the\_dataset.txt` | — | — | Demographics for COPD, Smokers, Control (wide, multi-table format) |



\### Discovered Structure

\- Each sensor file has \*\*no header row\*\*; columns repeat in blocks of \*\*8\*\*, one block per breath sample.

\- Sensor order within a block (per source paper): `SP-3, MQ-3, TGS822, MQ138, MQ137, TGS813, TGS800, MQ135`.

\- Validated empirically by checking that each sensor-position column has a consistent, characteristic value range across all samples in a file (rather than assuming the paper's stated layout blindly).



\### Key Decisions / Assumptions

\- \*\*SMOKERS group:\*\* 128 columns → 16 samples, but demographics only document 4 subjects. No reliable subject-to-sample mapping could be recovered, so each of the 16 samples is treated as \*\*independent\*\* — demographics are kept as a separate table, not merged 1:1 onto samples. GroupKFold/LOSO validation should exclude this group or fall back to sample-level grouping.

\- \*\*AIR group:\*\* has 80 rows beyond the expected 4000 timepoints — not truncated at ingestion (documented for downstream handling).

\- Sensor identity per column position is taken from the source paper's stated array order; it is \*\*not independently verifiable\*\* from the raw numeric files alone (no header row).



\### Data Quality Findings

\- No missing values in any raw file.

\- \*\*Exact duplicate block found in `COPD.txt`:\*\* subject 17's 2nd measurement (cols 264–271) is byte-identical to subject 18's 1st measurement (cols 272–279) — likely a copy-paste artifact in the source export. Flagged, not auto-removed (pending decision on whether to drop one).

\- No negative sensor values found (would have been clipped to 0 as implausible for MOS resistance readings).



\### Outputs

\- `demographics\_copd.csv`, `demographics\_control.csv`, `demographics\_smokers.csv` — cleaned per-group demographic tables.

\- `raw\_long\_<group>.parquet` — tidy long-format time series (`group, sample\_id, subject\_id, replicate, sensor, t, value`).

\- `sample\_index\_<group>.csv` / `sample\_index\_all.csv` — one row per sample with subject/replicate/demographics.

\- `ingestion\_report.md` — full write-up of parsing logic, validation checks, and caveats.



\### Folder Placement

\- \*\*raw/\*\* — `AIR.txt`, `CONTROL.txt`, `COPD.txt`, `SMOKERS.txt`, `General\_data\_from\_the\_dataset.txt`

\- \*\*interim/\*\* — `raw\_long\_\*.parquet`, `sample\_index\_\*.csv` (per-group, pre-consolidation)

\- \*\*processed/\*\* — `demographics\_\*.csv`, `sample\_index\_all.csv`

