import argparse
import os
import torch
import numpy as np
import pandas as pd

from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

# 기본 설정
parser.add_argument('--model', type=str, default='informer', help='model of experiment')
parser.add_argument('--data', type=str, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# 입력 시퀀스 설정
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# 모델 설정
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', default=True, help='use distilling')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention type')
parser.add_argument('--embed', type=str, default='timeF', help='embedding type')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='output attention')
parser.add_argument('--do_predict', action='store_true', help='predict unseen future data')
parser.add_argument('--mix', action='store_false', default=True, help='use mix attention')
parser.add_argument('--cols', type=str, nargs='+', help='input columns')

# 학습 설정
parser.add_argument('--num_workers', type=int, default=0, help='data loader workers')
parser.add_argument('--itr', type=int, default=2, help='experiment times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--des', type=str, default='test', help='experiment description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='lr adjustment')
parser.add_argument('--use_amp', action='store_true', default=False, help='use AMP')
parser.add_argument('--inverse', action='store_true', default=False, help='inverse output')

# GPU 설정
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='multi-gpu device ids')

args = parser.parse_args()

args.use_gpu = torch.cuda.is_available() and args.use_gpu

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7,7,7], 'S': [1,1,1], 'MS': [7,7,1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7,7,7], 'S': [1,1,1], 'MS': [7,7,1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7,7,7], 'S': [1,1,1], 'MS': [7,7,1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7,7,7], 'S': [1,1,1], 'MS': [7,7,1]},
    'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12,12,12], 'S': [1,1,1], 'MS': [12,12,1]},
    'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321,321,321], 'S': [1,1,1], 'MS': [321,321,1]},
    'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137,137,137], 'S': [1,1,1], 'MS': [137,137,1]},
}
if args.data in data_parser:
    info = data_parser[args.data]
    args.data_path = info['data']
    args.target = info['T']
    args.enc_in, args.dec_in, args.c_out = info[args.features]

args.s_layers = [int(s) for s in args.s_layers.split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer

for ii in range(args.itr):
    setting = f"{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_at{args.attn}_fc{args.factor}_eb{args.embed}_dt{args.distil}_mx{args.mix}_{args.des}_{ii}"

    exp = Exp(args)
    print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
    exp.train(setting)

    print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    exp.test(setting)

    if args.do_predict:
        print(f">>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        preds = exp.predict(setting, True)

        # ✅ 예측 결과 저장
        if preds is not None:
            preds = np.concatenate(preds, axis=0)
            reshaped = preds.reshape(-1, preds.shape[-1])
            df_pred = pd.DataFrame(reshaped)

            os.makedirs("./results", exist_ok=True)
            file_path = f"./results/predict_{setting}.csv"
            df_pred.to_csv(file_path, index=False)
            print(f"✅ 예측 결과 CSV 저장 완료: {file_path}")

    torch.cuda.empty_cache()
