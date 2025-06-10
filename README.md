# INFORMER

📦 **informer_project**  
├── 📂checkpoints/ — *모델 학습 결과 (pth 파일 등) 저장 폴더*  
│  
├── 📂data/ — *학습 및 검증용 센서 CSV 데이터*  
│   ├── 📄Train.csv  
│   └── 📄Valid.csv  
│  
├── 📂Informer2020/ — *Informer 원본 모델 코드 (하위 모듈 포함)*  
│   ├── 📂data/  
│   ├── 📂exp/ — *실험 실행 관련 파일 (로그, 결과 등)*  
│   ├── 📂models/ — *Informer 모델 구성 요소*  
│   │   ├── 📄attn.py — *ProbSparse attention 정의*  
│   │   ├── 📄decoder.py — *디코더 레이어 구성*  
│   │   ├── 📄embed.py — *임베딩 정의*  
│   │   ├── 📄encoder.py — *인코더 정의*  
│   │   ├── 📄model.py — *Informer 전체 모델 구조*  
│   │   └── 📄__init__.py  
│   ├── 📂utils/ — *유틸리티 함수 모음*  
│   │   ├── 📄masking.py — *마스킹 관련 유틸*  
│   │   ├── 📄metrics.py — *MSE, MAE 등 평가지표 계산*  
│   │   ├── 📄timefeatures.py — *시간 특성 추출 (주기 인식 등)*  
│   │   ├── 📄tools.py — *로깅, 설정, seed 고정 등*  
│   │   └── 📄__init__.py  
│   ├── 📂scripts/ — *모델 학습/실험용 외부 실행 스크립트 (사용 여부에 따라 생략 가능)*  
│   ├── 📄main_informer.py — *Informer 학습/예측 진입점 (CLI 기반 실행용)*  
│   └── 📄requirements.txt — *원본 모델용 의존성 목록*  
│  
├── 📂output/ — *예측 결과 .csv, .npy 등 저장 폴더*  
├── 📂results/ — *예측 결과 시각화 이미지, 로그 등*  
│  
├── 📄predict.py — *예측 결과 후처리 또는 상태 판단 스크립트*  
├── 📄train.py — *훈련 스크립트*  
├── 📄run_talkfile_informer.py — *실행 스크립트*  
└── 📄requirements.txt — *전체 프로젝트 의존성 목록*
