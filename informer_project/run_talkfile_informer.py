import argparse, subprocess, os, sys, shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR    = PROJECT_DIR / "Informer2020"
DATA_DIR    = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def preprocess(src: Path, dst: Path) -> int:
    print(f"📂 전처리 시작: {src}")
    df = pd.read_csv(src)
    print("✅ CSV 로드 완료, 컬럼 목록:", df.columns.tolist())
    if "date" not in df.columns:
        raise ValueError("'date' 컬럼이 존재하지 않습니다.")
    df = df.sort_values("date")
    drop_cols = [col for col in ["device_name", "collection_date", "collection_time"] if col in df.columns]
    df = df.drop(columns=drop_cols)
    df = df[["date"] + [c for c in df.columns if c != "date"]]
    df.to_csv(dst, index=False)
    print(f"💾 저장 완료: {dst}")
    return df.shape[1] - 1

def run(cmd: str):
    print(f"\n$ {cmd}\n" + "-"*80)
    subprocess.run(cmd, shell=True, check=True)

def main(args):
    train_csv = Path(args.train_csv)
    valid_csv = Path(args.valid_csv)

    enc_in = preprocess(train_csv, DATA_DIR / "Train.csv")
    _      = preprocess(valid_csv, DATA_DIR / "Valid.csv")

    PYTHON_EXE = sys.executable
    base = (
        f'"{PYTHON_EXE}" "{REPO_DIR}/main_informer.py" '
        f'--model informer '
        f'--data custom --root_path "{DATA_DIR}/" '
        f'--features M --freq t '
        f'--seq_len 36 --label_len 18 --pred_len 18 '
        f'--enc_in {enc_in} --dec_in {enc_in} --c_out {enc_in} '
        f'--attn prob --target state '
    )

    run(base + '--data_path Train.csv --batch_size 64 --train_epochs 30 --des talkfile_train')

    ckpt_dir = REPO_DIR / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("talkfile_train*.pth"))
    if not ckpts:
        sys.exit("❌ 체크포인트(.pth) 파일을 찾을 수 없습니다!")
    ckpt = ckpts[-1]

    run(base + f'--data_path Valid.csv --do_predict --checkpoints "{ckpt}" --des talkfile_infer')

def plot_prediction_result():
    result_dir = REPO_DIR / "results"
    result_file = sorted(result_dir.glob("predict_*.csv"))[-1]
    df_pred = pd.read_csv(result_file)
    df_valid = pd.read_csv(DATA_DIR / "Valid.csv")

    N = df_pred.shape[0]
    y_true = df_valid.iloc[-N:]["state"].reset_index(drop=True)
    y_pred = df_pred.iloc[:, 0]

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Actual (state)", linewidth=2)
    plt.plot(y_pred, label="Predicted", linestyle="--", linewidth=2)
    plt.title("Informer (state)")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    output_path = PROJECT_DIR / "output"
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / "prediction_plot_for_client.png", bbox_inches='tight')
    print(f"📊 예측 그래프 저장 완료: {output_path / 'prediction_plot_for_client.png'}")

def copy_prediction_result():
    result_dir = REPO_DIR / "results"
    result_file = sorted(result_dir.glob("predict_*.csv"))[-1]
    output_path = PROJECT_DIR / "output"
    output_path.mkdir(exist_ok=True)
    dst = output_path / "predict_result_for_client.csv"
    shutil.copy(result_file, dst)
    print(f"✅ 예측 결과 저장 완료: {dst}")

if __name__ == "__main__":
    print("🔥 main() 진입")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, help="학습용 CSV 파일 경로")
    parser.add_argument("--valid_csv", required=True, help="검증용 CSV 파일 경로")
    args = parser.parse_args()
    main(args)
    copy_prediction_result()
    copy_prediction_result()
    plot_prediction_result()
