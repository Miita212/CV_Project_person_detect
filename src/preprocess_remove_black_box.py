import os
import shutil
import cv2
import numpy as np

# 1. 경로 설정
img_dir = "person_data/images"
lbl_dir = "person_data/labels"
trash_img_dir = "person_data/corrupt_backup/images"
trash_lbl_dir = "person_data/corrupt_backup/labels"

os.makedirs(trash_img_dir, exist_ok=True)
os.makedirs(trash_lbl_dir, exist_ok=True)

valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
removed_count = 0

for img_name in os.listdir(img_dir):
    if not img_name.lower().endswith(valid_exts):
        continue

    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # 완전 검정색(픽셀값 5 이하) 마스크 생성
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_mask = (gray <= 5).astype(np.uint8) * 255

    # 검정 영역의 외곽선 검출
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    has_black_box = False
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # 노이즈가 아닌 일정 크기(예: 30x30 이상)의 사각형 형태 필터링
        if area > 900:
            extent = float(cv2.contourArea(cnt)) / area
            if extent > 0.85:  # 85% 이상 꽉 찬 직사각형일 때
                has_black_box = True
                break

    # 검정 박스가 감지되면 이미지와 해당 라벨을 격리
    if has_black_box:
        base_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(lbl_dir, base_name + ".txt")

        shutil.move(img_path, os.path.join(trash_img_dir, img_name))
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(trash_lbl_dir, base_name + ".txt"))
        
        removed_count += 1

print(f"검정 마킹 이미지 {removed_count}장 격리 완료.")