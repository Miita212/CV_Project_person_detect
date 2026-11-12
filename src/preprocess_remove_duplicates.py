import os
import shutil
import re
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 경로 설정 (본인 폴더 경로 확인)
# ==========================================
img_dir = "person_data/images"
lbl_dir = "person_data/labels"
dup_img_dir = "person_data/duplicate_backup/images"
dup_lbl_dir = "person_data/duplicate_backup/labels"

os.makedirs(dup_img_dir, exist_ok=True)
os.makedirs(dup_lbl_dir, exist_ok=True)

# ==========================================
# 2. 파일 자연 정렬 (frame_2 -> frame_10 순)
# ==========================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
file_list = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)], key=natural_sort_key)

print(f"총 검사 대상 이미지: {len(file_list)}장")

# ==========================================
# 3. 유사 프레임 필터링 실행
# ==========================================
MSE_THRESHOLD = 700.0  # 1200~1400대 정체 프레임 타깃 임계값
prev_img = None
removed_count = 0

for img_name in tqdm(file_list, desc="유사 프레임 정리 중"):
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # 미세 파도 거품 노이즈 완화를 위한 블러 + 64x64 리사이즈
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    curr_small = cv2.resize(blurred, (64, 64))

    # 첫 프레임 기준점 등록
    if prev_img is None:
        prev_img = curr_small
        continue

    # 기준 프레임과의 MSE(평균 제곱 오차) 계산
    mse = np.mean((prev_img.astype(np.float32) - curr_small.astype(np.float32)) ** 2)

    # 차이가 기준값 이하인 경우: 정체/중복 프레임으로 판정해 격리
    if mse < MSE_THRESHOLD:
        base_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(lbl_dir, base_name + ".txt")

        shutil.move(img_path, os.path.join(dup_img_dir, img_name))
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(dup_lbl_dir, base_name + ".txt"))
        
        removed_count += 1
        # 중복 이미지는 건너뛰고, 기준 프레임(prev_img)은 그대로 유지하여 다음 장과 비교
        continue

    # 의미 있는 시각적 변화가 감지되었을 때만 새 기준 프레임으로 갱신
    prev_img = curr_small

print("-" * 40)
print(f"작업 완료!")
print(f"격리된 유사 이미지: 총 {removed_count}장")
print(f"남은 정상 이미지: 총 {len(file_list) - removed_count}장")
print(f"격리 보관 위치: {dup_img_dir}")
print("-" * 40)