# src/train.py

from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from preprocess import load_data, add_binary_target, get_features_and_target, split_data


def train_model():
    print("📌 Veri yükleniyor...")
    df = load_data()

    print("📌 Binary hedef değişken ekleniyor...")
    df = add_binary_target(df)

    print("📌 X ve y ayrılıyor...")
    X, y = get_features_and_target(df)

    print("📌 Eğitim ve test setleri oluşturuluyor...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("📌 Model eğitiliyor (RandomForest)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("📌 Tahminler değerlendiriliyor...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n🎯 MODEL SONUÇLARI")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

    # Proje kök klasörünü bul
    root_dir = Path(__file__).resolve().parents[1]
    model_path = root_dir / "models" / "final_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    print(f"✅ Model kaydedildi: {model_path}")


if __name__ == "__main__":
    train_model()
