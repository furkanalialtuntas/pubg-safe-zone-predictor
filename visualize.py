import sys
import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold

from utils import load_data, apply_boundary_clamp, euclidean_distances
from constants import PHASE_RADIUS

DATA_PATH   = "data_augmented.csv"
MODEL_PATH  = "model.pkl"
MAP_PATH    = os.path.join("reference", "erangel_reference.png")
OUTPUT_NAME = "result_map.png"
NUM_SAMPLES = 1
N_SPLITS    = 5
PANEL_SIZE  = 700


def draw_sample(
    img_base: np.ndarray,
    test_input: np.ndarray,
    real_output: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    img = img_base.copy()

    current_phase = int(test_input[0])
    current_x     = int(test_input[1])
    current_y     = int(test_input[2])
    current_r     = int(test_input[3])

    real_next_x = int(real_output[0])
    real_next_y = int(real_output[1])

    next_phase = current_phase + 1
    next_r = PHASE_RADIUS.get(next_phase, PHASE_RADIUS[4])

    pred_x, pred_y, _ = apply_boundary_clamp(
        float(prediction[0]), float(prediction[1]),
        current_x, current_y, current_r, next_r,
    )

    # Beyaz: mevcut alan
    cv2.circle(img, (current_x, current_y), current_r, (255, 255, 255), 3)
    cv2.putText(
        img, f"Faz {current_phase}",
        (current_x - 30, current_y - current_r - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    # Yeşil: gerçek sonraki alan
    cv2.circle(img, (real_next_x, real_next_y), next_r, (0, 255, 0), 3)

    # Kırmızı: model tahmini
    cv2.circle(img, (pred_x, pred_y), next_r, (0, 0, 255), 3)

    # Sarı: hata çizgisi
    cv2.line(img, (real_next_x, real_next_y), (pred_x, pred_y), (0, 255, 255), 2)

    crop = int(current_r * 1.6)
    h, w = img.shape[:2]
    y1, y2 = max(0, current_y - crop), min(h, current_y + crop)
    x1, x2 = max(0, current_x - crop), min(w, current_x + crop)
    return cv2.resize(img[y1:y2, x1:x2], (PANEL_SIZE, PANEL_SIZE))


def main() -> None:
    if not os.path.exists(MAP_PATH):
        print(f"HATA: Erangel haritası '{MAP_PATH}' bulunamadı!")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası '{MODEL_PATH}' bulunamadı!")
        print("  → Önce 'python training.py' çalıştırın.")
        sys.exit(1)

    if not os.path.exists(DATA_PATH):
        print(f"HATA: Veri dosyası '{DATA_PATH}' bulunamadı!")
        print("  → Önce 'python augment_data.py' çalıştırın.")
        sys.exit(1)

    print(f"Model yükleniyor: '{MODEL_PATH}'")
    model = joblib.load(MODEL_PATH)

    print(f"Veri yükleniyor: '{DATA_PATH}'")
    X, Y, groups = load_data(DATA_PATH)

    gkf = GroupKFold(n_splits=N_SPLITS)
    _, test_idx = list(gkf.split(X, Y, groups))[-1]
    X_test, y_test = X[test_idx], Y[test_idx]
    print(f"Test seti: {len(X_test)} örnek")

    predictions = model.predict(X_test)

    rng = np.random.default_rng(seed=0)
    sample_indices = rng.choice(len(X_test), size=NUM_SAMPLES, replace=False)

    img_base = cv2.imread(MAP_PATH)
    panels = []

    print(f"\nGörsel hazırlanıyor ({NUM_SAMPLES} örnek)...")
    for idx in sample_indices:
        panels.append(draw_sample(img_base, X_test[idx], y_test[idx], predictions[idx]))

    cv2.imwrite(OUTPUT_NAME, cv2.vconcat(panels))
    print(f"\n✓ Sonuç kaydedildi: '{OUTPUT_NAME}'")


if __name__ == "__main__":
    main()
