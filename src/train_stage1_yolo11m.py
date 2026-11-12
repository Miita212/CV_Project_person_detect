"""
1차 학습: YOLO11m 사람 탐지 모델 (Roboflow 'search and rescue' 정제 데이터셋)

- 실제 학습 설정은 results/stage1_yolo11m_1024/args.yaml 에 기록된 값과 동일합니다.
- 데이터 경로(person_data/data.yaml)는 데이터셋 위치에 맞게 바꿔서 사용하세요.
- train / val 분할은 data.yaml 에 지정된 폴더(train, valid)를 그대로 사용합니다.
"""
import torch
from ultralytics import YOLO


def main():
    # RTX 4080 SUPER 가속 설정
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = YOLO("yolo11m.pt")

    model.train(
        data="person_data/data.yaml",
        epochs=40,
        patience=10,
        imgsz=1024,
        batch=8,        # 16 -> 8로 하향 (초기 메모리 할당 지연 및 화면 멈춤 방지)
        workers=0,      # Windows 멀티프로세스 충돌 방지
        cache=False,

        # 작은 객체 보호 및 박스 밀착 파라미터
        box=8.5,
        dfl=1.8,
        cls=1.0,
        mosaic=0.0,     # 모자이크 증강 끔 (작은 사람이 더 작아지는 것 방지)
        scale=0.15,

        lr0=0.001,
        lrf=0.01,
        device=0,
        name="yolo11m_clean_train_b8",
        save=True,
    )


if __name__ == "__main__":
    main()
