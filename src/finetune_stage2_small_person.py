"""
2차 파인튜닝: CCTV 화면 속 '작은 사람' 전용 데이터셋으로 1차 모델 미세 조정

- 1차 학습 best.pt 를 시작 가중치로 사용합니다.
- 실제 학습 설정은 results/stage2_finetune_1280/args.yaml 에 기록된 값과 동일합니다.
- 경로는 데이터셋·가중치 위치에 맞게 바꿔서 사용하세요.
"""
import torch
from ultralytics import YOLO


def main():
    # RTX 4080 SUPER 가속 설정
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 1차 학습 최고 가중치에서 시작
    model = YOLO("runs/detect/yolo11m_clean_train_b8/weights/best.pt")

    model.train(
        data="surfer_dataset/data_finetune.yaml",
        epochs=30,
        patience=8,

        # 먼 바다의 점처럼 작은 객체를 살리기 위한 고해상도
        imgsz=1280,
        batch=8,          # 16GB VRAM 에서 안정적으로 돌아가는 배치
        workers=8,
        cache="disk",

        # 작은 객체가 사라지지 않도록 증강 최소화
        mosaic=0.0,       # 1/4 축소로 작은 사람이 1픽셀이 되는 것 방지
        mixup=0.0,
        erasing=0.0,      # 머리·상체 가림 방지
        scale=0.1,

        # 미탐지를 줄이기 위한 손실 가중치
        cls=1.5,
        box=8.5,
        dfl=1.8,

        # 기존 가중치를 망가뜨리지 않는 낮은 학습률
        lr0=0.0008,
        lrf=0.01,
        freeze=0,
        rect=False,
        name="finetune_small_person_4080s",
        device=0,
        save=True,
    )


if __name__ == "__main__":
    main()
