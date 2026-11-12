"""
영상 추론 테스트: 학습한 사람 탐지 모델로 영상 속 입수자를 ByteTrack 으로 추적합니다.

사용법:
    python inference/track_person_video.py --weights best.pt --source test.mp4

※ 실제 서비스(팀 레포)에서는 이 스크립트 대신 SAHI 타일 슬라이싱(768x768, overlap 18%)으로
  먼 거리의 작은 사람을 검출합니다. 이 파일은 모델 단독 성능을 확인하기 위한 테스트용입니다.
"""
import argparse
import os
import time

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="best.pt", help="학습된 가중치 경로")
    p.add_argument("--source", default="test.mp4", help="입력 영상 경로")
    p.add_argument("--output", default="output_tracked_person.mp4", help="결과 영상 경로")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--device", default="cuda:0", help="GPU 가 없으면 cpu")
    p.add_argument("--show", action="store_true", help="추론 화면 띄우기")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # persist=True: 프레임 간 같은 사람에게 같은 ID 유지
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.5,
            device=args.device,
            verbose=False,
        )
        annotated = results[0].plot()

        now = time.time()
        cur_fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now
        ids = results[0].boxes.id
        tracked = 0 if ids is None else len(ids)

        cv2.rectangle(annotated, (15, 15), (420, 85), (0, 0, 0), -1)
        cv2.putText(annotated, f"Tracked Persons : {tracked}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {cur_fps:.1f} | Frame: {frame_count}", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        out.write(annotated)
        if args.show:
            cv2.imshow("Person Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"완료: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
