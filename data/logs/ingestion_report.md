# Phase 1 — Data Ingestion Report

Source: *Electronic nose dataset for COPD detection from smokers and healthy people through exhaled breath analysis* (Durán Acevedo et al., 2021, Data in Brief; Mendeley Data DOI 10.17632/h5pcn99zw4).

Sensor array (column order within each 8-col block): SP3, MQ3, TGS822, MQ138, MQ137, TGS813, TGS800, MQ135


## COPD
- Raw shape: 4000 timepoints x 320 columns
- Parsed as: 40 samples x 8 sensors
- Subject linkage: 20 subjects x 2 replicate measurements each (matches demographics file)
- Demographics rows available: 20
- Sensor-position sanity check (mean value per sensor slot, averaged across all samples in this file): SP3=0.380, MQ3=0.646, TGS822=0.777, MQ138=0.792, MQ137=0.812, TGS813=0.127, TGS800=0.267, MQ135=0.516

## CONTROL
- Raw shape: 4000 timepoints x 160 columns
- Parsed as: 20 samples x 8 sensors
- Subject linkage: 10 subjects x 2 replicate measurements each (matches demographics file)
- Demographics rows available: 10
- Sensor-position sanity check (mean value per sensor slot, averaged across all samples in this file): SP3=0.631, MQ3=1.116, TGS822=1.087, MQ138=1.434, MQ137=1.360, TGS813=0.224, TGS800=0.362, MQ135=0.987

## SMOKERS
- Raw shape: 4000 timepoints x 128 columns
- Parsed as: 16 samples x 8 sensors
- Subject linkage: NOT assumed. 16 samples treated as independent (per explicit decision) — demographics (4 subjects) NOT merged 1:1 onto samples. See demographics_smokers.csv separately.
- Sensor-position sanity check (mean value per sensor slot, averaged across all samples in this file): SP3=0.789, MQ3=1.233, TGS822=1.793, MQ138=1.371, MQ137=1.228, TGS813=0.205, TGS800=0.672, MQ135=0.945

## AIR
- Raw shape: 4080 timepoints x 80 columns
- Parsed as: 10 samples x 8 sensors
- No associated subject/demographics (ambient air baseline).
- ⚠️ Row count is 4080, not the expected 4000 — 80 extra rows kept as-is; verify against source before using for fixed-length feature windows.
- Sensor-position sanity check (mean value per sensor slot, averaged across all samples in this file): SP3=0.534, MQ3=1.839, TGS822=0.329, MQ138=1.676, MQ137=1.738, TGS813=0.303, TGS800=0.664, MQ135=1.793

## Combined sample index
- Total samples across all groups: 86
group
air        10
control    20
copd       40
smokers    16

## Caveats / assumptions carried into Phase 2
- Sensor identity assignment (SP-3, MQ-3, TGS822, MQ138, MQ137, TGS813, TGS800, MQ135, in that column order) follows the source paper's stated array order — it is NOT independently verifiable from the raw numeric files alone, since there is no header row. Values are unitless in the raw text (paper describes them as voltage-domain sensor responses).
- SMOKERS group: sample-to-subject mapping is unresolved (16 raw samples vs 4 documented subjects). Demographics for this group are kept as a separate table, not joined to samples. Any modeling that needs GroupKFold/LOSO on smokers should either exclude this group or use sample_id as a (weaker) grouping key.
- AIR group has 80 extra rows relative to the other three files (4080 vs 4000) - not truncated here.