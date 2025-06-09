import argparse, subprocess, sys
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR    = PROJECT_DIR / "Informer2020"
DATA_DIR    = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def preprocess(src: Path, dst: Path) -> int:
    print(f"📂 전처리 시작: {src}")
    df = pd.read_csv(src)
    print("✅ CSV 로드 완료, 콜런 목록:", df.columns.tolist())
    if "date" not in df.columns:
        raise ValueError("'date' 콜런이 존재하지 않습니다.")
    df = df.sort_values("date")
    drop_cols = [col for col in ["device_name", "collection_date", "collection_time"] if col in df.columns]
    df = df.drop(columns=drop_cols)
    df = df[["date"] + [c for c in df.columns if c != "date"]]
    df.to_csv(dst, index=False)
    print(f"📂 저장 완료: {dst}")
    return df.shape[1] - 1

def run(cmd: str):
    print(f"\n$ {cmd}\n" + "-"*80)
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    print("🔥 학습 시작")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    args = parser.parse_args()

    train_csv = Path(args.train_csv)
    enc_in = preprocess(train_csv, DATA_DIR / "Train.csv")

    PYTHON_EXE = sys.executable
    cmd = (
        f'"{PYTHON_EXE}" "{REPO_DIR}/main_informer.py" '
        f'--model informer --data custom --root_path "{DATA_DIR}/" '
        f'--features M --freq t --seq_len 36 --label_len 18 --pred_len 18 '
        f'--enc_in {enc_in} --dec_in {enc_in} --c_out {enc_in} '
        f'--attn prob --target state '
        f'--data_path Train.csv --batch_size 64 --train_epochs 1 --itr 1 --des talkfile_train'
    )
    run(cmd)
    