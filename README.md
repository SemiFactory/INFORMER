# INFORMER
📦informer_project
├── 📂checkpoints/                  # 모델 학습 결과(pth 파일 등) 저장 폴더
│
├── 📂data/                         # 학습 및 검증용 CSV 데이터
│   ├── Train.csv
│   └── Valid.csv
│
├── 📂Informer2020/                # Informer 원본 모델 코드 (하위 모듈 포함)
│   ├── 📂data/                    
│   ├── 📂exp/                      # 실험 실행 관련 파일 
│   ├── 📂models/                   # Informer 모델 구성 요소
│   │   ├── attn.py                # ProbSparse attention 등 정의
│   │   ├── decoder.py             # 디코더 레이어 구성
│   │   ├── embed.py               # 임베딩 정의
│   │   ├── encoder.py             # 인코더 정의
│   │   ├── model.py               # Informer 모델 전체 구조
│   │   └── __init__.py
│   │
│   ├── 📂utils/                    # 유틸리티 함수들
│   │   ├── masking.py             # 마스킹 유틸
│   │   ├── metrics.py             # MSE, MAE 등 지표 계산
│   │   ├── timefeatures.py        # 시간 특징 추출 (시간대별 주기 인식)
│   │   ├── tools.py               # 설정, 로깅, seed 설정 등
│   │   └── __init__.py
│   │
│   ├── 📂scripts/                  # 실험 스크립트 
│   ├── main_informer.py           # Informer 학습/테스트 진입점 (CLI 기반)
│   └── requirements.txt           # 원본 코드용 의존성
│
├── 📂output/                       # 예측 결과 파일 저장 폴더
│
├── 📂results/                      # 예측 결과 이미지, 로그 등 저장 폴더
│
├── predict.py                     # 후처리 또는 커스텀 예측 실행 스크립트
├── train.py                       # 훈련 실행 스크립트 
├── run_talkfile_informer.py       # 실행 스크립트
└── requirements.txt               # 전체 프로젝트 의존성 목록
