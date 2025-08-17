import argparse
import ast
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Arquivo icd_list_source_v10.csv original")
    parser.add_argument("--output", required=True, help="Arquivo de saída com colunas stay_id e icd_code")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # criar lista de linhas (stay_id, icd_code)
    rows = []
    for _, row in df.iterrows():
        stay_id = row["stay_id"]
        # ast.literal_eval converte a string "{'A', 'B'}" em set
        icd_set = set()
        if isinstance(row["icd_list"], str) and row["icd_list"] != "set()":
            icd_set = ast.literal_eval(row["icd_list"])
        for code in icd_set:
            rows.append({"stay_id": stay_id, "icd_code": code})
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(f"Arquivo gerado em {args.output} com {len(result)} linhas.")

if __name__ == "__main__":
    main()
