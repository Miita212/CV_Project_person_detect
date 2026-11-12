"""
학습에 쓰지 않은 해안 서프캠 데이터(my_dataset/val, 2,413장)로
1차 모델과 2차 파인튜닝 모델을 다시 평가하는 스크립트.

준비물 (이 파일과 같은 폴더 기준):
  weights/stage1_best.pt   <- 노트북 runs/detect/rip_current_project/yolo11m_clean_train_b8/weights/best.pt
  weights/stage2_best.pt   <- 노트북 runs/detect/rip_current_project/finetune_small_person_4080s-2/weights/best.pt
  ../CV11_Project.zip      <- 이미 rip_current 폴더에 있음

실행:
  conda activate rip_current
  python eval_unseen.py
"""
import json
import zipfile
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ZIP_PATH = HERE.parent / "CV11_Project.zip"
DATA_DIR = HERE / "unseen_test"
YAML_PATH = DATA_DIR / "test.yaml"

MODELS = {
    "stage1_yolo11m_1024": HERE / "weights" / "stage1_best.pt",
    "stage2_finetune_1280": HERE / "weights" / "stage2_best.pt",
}
IMG_SIZES = [1024, 1280]


def extract_val_split():
    """zip 에서 my_dataset 의 val 이미지·라벨만 꺼낸다 (약 1.2GB)."""
    if (DATA_DIR / "images" / "test").exists():
        print("테스트셋이 이미 있어 압축 해제를 건너뜁니다.")
        return
    print("zip 에서 val 이미지·라벨을 꺼내는 중...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        for name in z.namelist():
            for kind in ("images", "labels"):
                prefix = f"my_dataset/{kind}/val/"
                if name.startswith(prefix) and not name.endswith("/"):
                    out = DATA_DIR / kind / "test" / name[len(prefix):]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))
    print("완료")


def write_yaml():
    # 원래 클래스 이름은 swimmer 이지만 id 0 하나뿐이라 person 모델과 그대로 비교 가능
    YAML_PATH.write_text(
        f"path: {DATA_DIR.as_posix()}\n"
        "train: images/test\n"
        "val: images/test\n"
        "test: images/test\n"
        "names:\n  0: person\n",
        encoding="utf-8",
    )


def main():
    for name, w in MODELS.items():
        if not w.exists():
            raise SystemExit(f"가중치가 없습니다: {w}")

    extract_val_split()
    write_yaml()

    rows = []
    for name, w in MODELS.items():
        model = YOLO(str(w))
        for imgsz in IMG_SIZES:
            m = model.val(
                data=str(YAML_PATH),
                split="test",
                imgsz=imgsz,
                batch=8,
                conf=0.001,
                iou=0.6,
                device=0,
                plots=True,
                project=str(HERE / "runs"),
                name=f"{name}_img{imgsz}",
                exist_ok=True,
                verbose=False,
            )
            row = {
                "model": name,
                "imgsz": imgsz,
                "precision": round(float(m.box.mp), 4),
                "recall": round(float(m.box.mr), 4),
                "mAP50": round(float(m.box.map50), 4),
                "mAP50-95": round(float(m.box.map), 4),
            }
            rows.append(row)
            print(row)

    (HERE / "unseen_test_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 처음 보는 해안 서프캠 데이터 (2,413장) 평가 결과 ===")
    print(f"{'model':<24}{'imgsz':>6}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for r in rows:
        print(f"{r['model']:<24}{r['imgsz']:>6}{r['precision']:>8.3f}{r['recall']:>8.3f}"
              f"{r['mAP50']:>8.3f}{r['mAP50-95']:>10.3f}")
    print("\n결과 파일: unseen_test_results.json")


if __name__ == "__main__":
    main()
