import sys
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold

from utils import load_data, euclidean_distances
from constants import PHASE_RADIUS

DATA_PATH  = "data_augmented.csv"
MODEL_PATH = "model.pkl"
N_SPLITS   = 5

XGB_PARAMS = dict(
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 6,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    random_state      = 42,
    n_jobs            = -1,
    tree_method       = "hist",
)


def build_model() -> MultiOutputRegressor:
    return MultiOutputRegressor(XGBRegressor(**XGB_PARAMS))


def cross_validate(X: np.ndarray, Y: np.ndarray, groups: list) -> dict[str, float]:
    gkf = GroupKFold(n_splits=N_SPLITS)
    mae_x_list, mae_y_list, dist_list = [], [], []

    print(f"── GroupKFold Çapraz Doğrulama ({N_SPLITS} Fold) ──")
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = Y[tr_idx], Y[te_idx]

        model = build_model()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        mae_x = float(np.mean(np.abs(y_te[:, 0] - preds[:, 0])))
        mae_y = float(np.mean(np.abs(y_te[:, 1] - preds[:, 1])))
        dist  = float(euclidean_distances(y_te, preds).mean())

        mae_x_list.append(mae_x)
        mae_y_list.append(mae_y)
        dist_list.append(dist)

        print(f"  Fold {fold + 1}/{N_SPLITS}: "
              f"MAE_X={mae_x:.1f}px  MAE_Y={mae_y:.1f}px  "
              f"Ortalama Sapma={dist:.1f}px  "
              f"[Test: {len(X_te)} örnek]")

    results = {
        "mae_x_mean": float(np.mean(mae_x_list)),
        "mae_x_std":  float(np.std(mae_x_list)),
        "mae_y_mean": float(np.mean(mae_y_list)),
        "mae_y_std":  float(np.std(mae_y_list)),
        "dist_mean":  float(np.mean(dist_list)),
        "dist_std":   float(np.std(dist_list)),
    }
    print(f"\n── Genel CV Sonuçları ──")
    print(f"  Ortalama MAE_X  : {results['mae_x_mean']:.1f} ± {results['mae_x_std']:.1f} px")
    print(f"  Ortalama MAE_Y  : {results['mae_y_mean']:.1f} ± {results['mae_y_std']:.1f} px")
    print(f"  Ortalama Sapma  : {results['dist_mean']:.1f} ± {results['dist_std']:.1f} px")
    return results


def train_final_model(X: np.ndarray, Y: np.ndarray) -> MultiOutputRegressor:
    print("\nFinal model tüm veri üzerinde eğitiliyor...")
    model = build_model()
    model.fit(X, Y)
    return model


def sample_predictions(model: MultiOutputRegressor, X: np.ndarray, Y: np.ndarray, n: int = 5) -> None:
    rng = np.random.default_rng(seed=7)
    indices = rng.choice(len(X), size=min(n, len(X)), replace=False)
    preds = model.predict(X[indices])

    print(f"\n── Örnek Tahminler (n={n}) ──")
    for i, idx in enumerate(indices):
        inp   = X[idx]
        real  = Y[idx]
        pred  = preds[i]
        phase = int(inp[0])
        dist  = float(euclidean_distances(real[None], pred[None])[0])
        next_r = PHASE_RADIUS.get(phase + 1, PHASE_RADIUS[4])
        print(
            f"  [{i+1}] Faz {phase}→{phase+1}  "
            f"Mevcut:({int(inp[1])},{int(inp[2])})  "
            f"Gerçek:({int(real[0])},{int(real[1])})  "
            f"Tahmin:({int(pred[0])},{int(pred[1])})  "
            f"Sapma:{dist:.1f}px  "
            f"[R={PHASE_RADIUS[phase]}→{next_r}]"
        )


def main() -> None:
    print(f"Veri yükleniyor: {DATA_PATH}")
    try:
        X, Y, groups = load_data(DATA_PATH)
    except FileNotFoundError:
        print(f"HATA: '{DATA_PATH}' bulunamadı. Önce 'python augment_data.py' çalıştırın.")
        sys.exit(1)

    print(f"Toplam {len(X)} eğitim çifti  ({len(set(groups))} maç)\n")

    cross_validate(X, Y, groups)
    model = train_final_model(X, Y)
    sample_predictions(model, X, Y)

    joblib.dump(model, MODEL_PATH)
    print(f"\n✓ Model kaydedildi: '{MODEL_PATH}'")
    print("  (Görselleştirme için 'python visualize.py' çalıştırabilirsiniz.)")


if __name__ == "__main__":
    main()
