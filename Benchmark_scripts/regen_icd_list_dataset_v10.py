# regen_icd_list_dataset_v10.py
import os
import argparse
import pandas as pd
import numpy as np

def norm_icd(code: str) -> str:
    """Normaliza ICD: maiúscula, remove pontos, remove espaços."""
    if pd.isna(code):
        return None
    s = str(code).strip().upper().replace('.', '')
    return s or None

def load_vocab_or_build(df_icd_raw, vocab_csv, version='v10', max_size=None):
    """
    Carrega vocabulário oficial (CSV com icd_norm,idx) OU constrói a partir dos dados.
    Se construir, idxs = [0..N-1] por ordem de frequência (ou alfabética).
    """
    if vocab_csv and os.path.exists(vocab_csv):
        vocab = pd.read_csv(vocab_csv)
        assert {'icd_norm','idx'}.issubset(vocab.columns), \
            "Vocabulário precisa de colunas: icd_norm, idx"
        vocab = vocab.dropna(subset=['icd_norm','idx'])
        vocab['idx'] = vocab['idx'].astype(int)
        vocab = vocab[['icd_norm','idx']].drop_duplicates()
        vocab_size = int(vocab['idx'].max()) + 1
        vocab_map = dict(zip(vocab['icd_norm'], vocab['idx']))
        print(f"[OK] Vocabulário carregado: size={vocab_size}")
        return vocab_map, vocab_size, 'official'
    else:
        print("[WARN] Vocabulário oficial não fornecido. Construindo a partir dos dados…")
        # conta frequência por icd_norm
        counts = (df_icd_raw[['icd_norm']]
                  .dropna()
                  .value_counts()
                  .reset_index(name='freq'))
        counts.rename(columns={'icd_norm': 'icd_norm'}, inplace=True)
        # se max_size for dado, keep top-k
        if max_size is not None:
            counts = counts.sort_values('freq', ascending=False).head(max_size)
        counts = counts.reset_index(drop=True)
        counts['idx'] = np.arange(len(counts), dtype=int)
        vocab_map = dict(zip(counts['icd_norm'], counts['idx']))
        vocab_size = len(vocab_map)
        print(f"[OK] Vocabulário construído: size={vocab_size}")
        return vocab_map, vocab_size, 'built'

def build_icd_list_by_stay(df_icd_raw, vocab_map, allow_oov=True):
    """
    Gera uma lista de índices por stay_id:
      - se allow_oov=True, códigos fora do vocabulário são marcados como OOV (usaremos no gerador).
      - se allow_oov=False, remove códigos fora do vocabulário (conformidade estrita).
    """
    rows = []
    oov_ct = 0
    tot_ct = 0
    for sid, grp in df_icd_raw.groupby('stay_id'):
        idxs = []
        for _, r in grp.iterrows():
            tot_ct += 1
            icd = r['icd_norm']
            if icd is None: 
                continue
            if icd in vocab_map:
                idxs.append(int(vocab_map[icd]))  # 0..(vocab_size-1)
            else:
                if allow_oov:
                    # marcamos como -2 agora; o gerador converterá OOV corretamente
                    idxs.append(-2)
                    oov_ct += 1
                # se allow_oov=False, apenas ignora
        # remover duplicatas por stay (opcional) e ordenar
        idxs = sorted(set(idxs))
        rows.append((sid, idxs))
    df = pd.DataFrame(rows, columns=['stay_id','icd_encoded_list'])
    print(f"[INFO] Total códigos={tot_ct} | OOV encontrados={oov_ct} ({oov_ct/max(tot_ct,1):.2%})")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True,
                    help="Diretório onde salvar/ler arquivos (mesmo 'path' do notebook).")
    ap.add_argument("--icd_source_csv", required=True,
                    help="CSV com colunas: stay_id, icd_code (1 linha por código por stay).")
    ap.add_argument("--vocab_csv", default=None,
                    help="CSV opcional com colunas: icd_norm, idx (vocabulário oficial v10).")
    ap.add_argument("--version", default="v10", help="Rótulo da versão (ex: v10).")
    ap.add_argument("--strict", action="store_true",
                    help="Se setado, remove códigos fora do vocabulário (sem OOV).")
    ap.add_argument("--max_vocab_size", type=int, default=None,
                    help="Se construir vocabulário, limita ao top-k mais frequentes.")
    args = ap.parse_args()

    path = args.path
    os.makedirs(path, exist_ok=True)

    # 1) Carrega ICD bruto
    df_icd_raw = pd.read_csv(args.icd_source_csv)
    assert {'stay_id','icd_code'}.issubset(df_icd_raw.columns), \
        "Esperado: colunas stay_id, icd_code"
    df_icd_raw['icd_norm'] = df_icd_raw['icd_code'].map(norm_icd)
    df_icd_raw = df_icd_raw.dropna(subset=['icd_norm','stay_id'])

    # 2) Vocabulário (oficial ou construído)
    vocab_map, vocab_size, mode = load_vocab_or_build(
        df_icd_raw, args.vocab_csv, version=args.version, max_size=args.max_vocab_size
    )

    # 3) Gera listas de índices por stay_id
    df_icd_enc = build_icd_list_by_stay(
        df_icd_raw[['stay_id','icd_norm']], vocab_map, allow_oov=(not args.strict)
    )

    # 4) Salva no formato esperado pelo notebook
    out_csv = os.path.join(path, f"icd_list_dataset_{args.version}.csv")
    df_icd_enc.to_csv(out_csv, index=False)
    print(f"[OK] Salvo: {out_csv}")

    # 5) (Opcional) salva vocabulário efetivo usado, para documentação
    vocab_out = os.path.join(path, f"vocab_{args.version}_{mode}.csv")
    pd.DataFrame({"icd_norm": list(vocab_map.keys()), "idx": list(vocab_map.values())}) \
        .sort_values("idx").to_csv(vocab_out, index=False)
    print(f"[OK] Vocabulário efetivo salvo em: {vocab_out}")
    print(f"[HINT] Use vocab_size={vocab_size} no notebook se você estiver no modo '{mode}'.")
    
if __name__ == "__main__":
    main()
