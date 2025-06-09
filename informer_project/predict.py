import subprocess, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ✅ 사용자 지정 경로
PROJECT_DIR = Path("C:/Users/leedowon/OneDrive/바탕 화면/informer/informer_project")
REPO_DIR = PROJECT_DIR / "Informer2020"
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_DIR = PROJECT_DIR / "output"
VALID_CSV_PATH = DATA_DIR / "Valid.csv"
CKPT_PATH = PROJECT_DIR / "checkpoints/informer_custom_ftM_sl36_ll18_pl18_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_talkfile_train_0/checkpoint.pth"

# 폴더 확인
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 전처리
def preprocess(src: Path, dst: Path):
    print(f"📂 전처리 시작: {src}")
    df = pd.read_csv(src)
    df = df.sort_values("date")
    drop_cols = [col for col in ["device_name", "collection_date", "collection_time"] if col in df.columns]
    df = df.drop(columns=drop_cols)
    df = df[["date"] + [c for c in df.columns if c != "date"]]
    df.to_csv(dst, index=False)
    print(f"📂 저장 완료: {dst}")

# 커맨드 실행
def run(cmd: str):
    print(f"\n$ {cmd}\n" + "-"*80)
    subprocess.run(cmd, shell=True, check=True)

# 결과 시각화
def plot_prediction_result():
    result_files = sorted(RESULTS_DIR.glob("predict_*.csv"))
    if not result_files:
        print("❗ 예측 결과 CSV 파일을 찾을 수 없습니다.")
        return
    result_file = result_files[-1]
    df_pred = pd.read_csv(result_file)
    df_valid = pd.read_csv(VALID_CSV_PATH)
    df_true = df_valid.iloc[-df_pred.shape[0]:].reset_index(drop=True)

    df_pred.columns = df_pred.columns.astype(str)
    df_true.columns = df_true.columns.astype(str)

    for col in df_pred.columns:
        if col in df_true.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(df_true[col], label="Actual", linewidth=2)
            plt.plot(df_pred[col], label="Predicted", linestyle="--", linewidth=2)
            plt.title(f"Informer Prediction: {col}")
            plt.xlabel("Time step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plot_path = OUTPUT_DIR / f"prediction_{col}.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"📊 예측 그래프 저장 완료: {plot_path}")

# 예측 vs 실제 출력 (정규화 복원 포함)
def print_stepwise_prediction():
    result_files = sorted(RESULTS_DIR.glob("predict_*.csv"))
    if not result_files:
        print("❗ 예측 결과 CSV 파일을 찾을 수 없습니다.")
        return

    result_file = result_files[-1]
    df_pred = pd.read_csv(result_file)
    df_valid = pd.read_csv(VALID_CSV_PATH)

    df_pred.columns = df_pred.columns.astype(str)
    df_valid.columns = df_valid.columns.astype(str)

    if 'date' in df_valid.columns:
        df_valid = df_valid.drop(columns=['date'])

    df_valid = df_valid.iloc[-df_pred.shape[0]:].reset_index(drop=True)

    # 정규화 복원
    scaler = StandardScaler()
    scaler.fit(df_valid.values)
    pred_inv = scaler.inverse_transform(df_pred.values)
    true_inv = scaler.inverse_transform(df_valid.values)

    df_pred_inv = pd.DataFrame(pred_inv, columns=df_valid.columns)
    df_true_inv = pd.DataFrame(true_inv, columns=df_valid.columns)

    print("\n📋 센서별 예측 vs 실제 값 비교 (역정규화):")
    for i in range(min(10, len(df_pred_inv))):
        pred_row = df_pred_inv.iloc[i]
        true_row = df_true_inv.iloc[i]

        pred_str = ", ".join([f"{col}: {pred_row[col]:.2f}" for col in df_pred_inv.columns])
        true_str = ", ".join([f"{col}: {true_row[col]:.2f}" for col in df_pred_inv.columns])

        print(f"[Step {i}]")
        print(f"  예측 [{pred_str}]")
        print(f"  실제 [{true_str}]")

# 메인 실행
if __name__ == "__main__":
    print("🔍 예측 시작")

    preprocess(VALID_CSV_PATH, VALID_CSV_PATH)

    PYTHON = sys.executable
    enc_in = 13

    cmd = (
        f'"{PYTHON}" "{REPO_DIR}/main_informer.py" '
        f'--model informer --data custom --root_path "{DATA_DIR}/" '
        f'--features M --freq t --seq_len 36 --label_len 18 --pred_len 18 '
        f'--enc_in {enc_in} --dec_in {enc_in} --c_out {enc_in} '
        f'--attn prob --target state '
        f'--data_path Valid.csv --do_predict '
        f'--checkpoints "{CKPT_PATH.parent}" '
        f'--des talkfile_infer --itr 1'
    )

    run(cmd)
    plot_prediction_result()
    print_stepwise_prediction()
