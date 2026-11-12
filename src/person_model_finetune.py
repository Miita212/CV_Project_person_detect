import torch
from ultralytics import YOLO

def main():
    # 1. RTX 4080 SUPER 하드웨어 가속 설정
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 2. 기존 베스트 모델 불러오기 (파인튜닝 베이스)
    model = YOLO("C:/Users/Martin/Documents/CVProject/rip_current/runs/detect/yolo11m_clean_train_b8/weights/best.pt")

    # 3. 초소형 객체 미탐지 전용 파인튜닝 실행 (Train 9 : Val 1 분할 추가)
    results = model.train(
        data=r"C:/Users/Martin/Documents/CVProject/rip_current/surfer_dataset/data_finetune.yaml",
        split=0.9,           # Train 90%, Val 10% 비율로 자동 분할
        epochs=30,           # 짧고 굵게 미세 조정 (30 에포크면 충분)
        patience=8,          # 조기 종료 설정
        
        # ── [극소형 점 객체 보존 핵심] ──
        imgsz=1280,          # 먼바다 점 객체 해상도 유지 (4080s 16GB 여유 활용)
        batch=8,             # 화면 멈춤 없이 16GB VRAM을 안전하게 활용하는 최적 배치
        workers=8,           # CPU 데이터 로딩 병목 차단 (에포크당 1분대 완주)
        cache="disk",        # 안정적인 디스크 캐싱 (I/O 지연 제거)
        
        # ── [데이터 증강 차단 (점 객체 소실 방지)] ──
        mosaic=0.0,          # 1/4 축소 방지 (작은 사람이 1픽셀로 소실되는 것 차단)
        mixup=0.0,
        erasing=0.0,         # 머리/상체 가림 방지
        scale=0.1,           # 과도한 축소 방지
        
        # ── [미탐지 해결 손실 가중치] ──
        cls=1.5,             # 클래스 민감도 상향 (흐릿한 점 객체 반응 유도)
        box=8.5,             # 타이트한 박스 밀착
        dfl=1.8,
        
        # ── [학습률 안정화] ──
        lr0=0.0008,          # 기존 가중치 파괴를 막기 위한 낮은 초기 학습률
        lrf=0.01,
        freeze=0,            # 전체 레이어를 풀어 1280 해상도에 고르게 안착
        rect=False,
        name="finetune_small_person_4080s",
        device=0,
        save=True
    )

if __name__ == "__main__":
    main()