import torch
from ultralytics import YOLO

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

model = YOLO("yolo11m.pt")

results = model.train(
    data="person_data/data.yaml",
    epochs=40,
    patience=10,
    imgsz=1024,
    batch=8,           # 16 -> 8로 하향 (초기 메모리 할당 랙 및 화면 멈춤 차단)
    workers=0,         # 윈도우 프로세스 충돌 차단
    cache=False,
    
    # train / val 8:2 분할 설정
    fraction=0.8,      # 전체 데이터 중 80%만 학습/검증에 사용 (8:2 분할 적용)
    
    # 초소형 객체 보호 및 박스 밀착 파라미터 유지
    box=8.5,
    dfl=1.8,
    cls=1.0,
    mosaic=0.0,
    scale=0.15,
    
    lr0=0.001,
    lrf=0.01,
    device=0,
    name="yolo11m_clean_train_b8",
    save=True
)