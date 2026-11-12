import os
import time
import cv2
from ultralytics import YOLO

# 1. 학습 완료된 YOLO11 가중치 로드
# 학습 결과 경로(runs/detect/finetune_small_person_4080s-2/weights/best.pt 등) 또는 실제 파일 경로 지정
model_path = "runs/detect/rip_person_opt/weights/best.pt"
if not os.path.exists(model_path):
    model_path = "best.pt"  # 현재 폴더에 복사해 둔 경우 대체

model = YOLO(model_path)

# 2. 테스트할 입력/출력 영상 설정
input_video_path = "testdata/rip_current.mp4"  # 테스트할 비디오 파일 경로 (웹캠/CCTV는 0 또는 RTSP URL)
output_video_path = "output_tracked_person.mp4"

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"❌ 영상을 열 수 없습니다: {input_video_path}")
    exit()

# 비디오 스펙 확인
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30.0 if (fps == 0 or fps is None) else fps

# 결과 저장용 VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 윈도우 창 크기 설정 (화면 잘림 방지)
window_name = "YOLO11 Person Tracking (Rip-Current)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

print("🚀 영상 추론 시작! (종료: 영상 창 클릭 후 'q' 키)")

frame_count = 0
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 3. 실시간 트래킹 추론
    # - persist=True: 프레임 간 동일 객체 ID 유지
    # - imgsz=1024: 학습 해상도 유지하여 원거리/소형 입수자 포착
    # - conf=0.20: 파도 거품 오탐을 막으면서 물속 입수자를 놓치지 않는 신뢰도 기준
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=1024,
        conf=0.10,
        iou=0.5,
        device="cuda:0",  # GPU 가속 (없을 경우 'cpu')
        verbose=False
    )

    # 4. 시각화 프레임 생성
    annotated_frame = results[0].plot()

    # 실시간 FPS 계산
    curr_time = time.time()
    current_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # 현재 프레임 탐지된 인원 수 계산
    tracked_count = 0
    if results[0].boxes.id is not None:
        tracked_count = len(results[0].boxes.id)

    # 5. 상태창 오버레이 (인원 수, FPS, 프레임 번호)
    cv2.rectangle(annotated_frame, (15, 15), (420, 85), (0, 0, 0), -1)
    cv2.putText(
        annotated_frame,
        f"Tracked Swimmers: {tracked_count}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )
    cv2.putText(
        annotated_frame,
        f"FPS: {current_fps:.1f} | Frame: {frame_count}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1
    )

    # 6. 화면 출력 및 파일 저장
    cv2.imshow(window_name, annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n사용자에 의해 종료되었습니다.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ 완료! 결과 영상이 저장되었습니다: {os.path.abspath(output_video_path)}")