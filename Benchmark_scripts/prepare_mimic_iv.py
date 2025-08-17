import os, pathlib, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
ver = os.getenv("MIMIC_ED_VERSION", "1.0")
RAW = ROOT / "data" / "raw" / "mimic-iv-ed" / ver
OUT_I = ROOT / "data" / "interim" / "mimic-iv-ed" / ver
OUT_P = ROOT / "data" / "processed" / "mimic-iv-ed" / ver
OUT_I.mkdir(parents=True, exist_ok=True)
OUT_P.mkdir(parents=True, exist_ok=True)

# Exemplos de tabelas comuns do MIMIC-IV-ED:
# edstays.csv.gz, triage.csv.gz, vitalsign.csv.gz, diagnosis.csv.gz, medrecon.csv.gz
def load_csv(name):
    return pd.read_csv(RAW / f"{name}.csv.gz", low_memory=False)

edstays   = load_csv("edstays")
triage    = load_csv("triage")
vitals    = load_csv("vitalsign")
diagnosis = load_csv("diagnosis")

# Join básico (ajuste conforme seu pipeline)
base = (edstays
        .merge(triage, on=["stay_id"], how="left", suffixes=("","_tri"))
        .merge(diagnosis, on=["stay_id"], how="left", suffixes=("","_dx")))

# Exemplo de recorte: apenas colunas úteis ao triage/ML inicial
cols = [c for c in base.columns if c in {
    "subject_id","stay_id","charttime","intime","outtime","age","gender",
    "acuity","temperature","heartrate","resprate","o2sat","sbp","dbp",
    "chiefcomplaint","edregtime","edouttime","disposition","icd_code","icd_version"
}]
subset = base[cols].copy()

subset.to_parquet(OUT_P / "triage_baseline.parquet", index=False)
print("OK ->", OUT_P / "triage_baseline.parquet")
