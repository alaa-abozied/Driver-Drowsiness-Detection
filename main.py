"""
============================================================
  CNN-LSTM Driver Drowsiness Detection
  Project: Deep Learning Assignment
  Model: CNN-LSTM Hybrid Architecture
  Dataset: Synthetic frame sequences (or custom NPZ)
============================================================

DESCRIPTION:
  This project follows the same style used in the HAR/SLR solutions,
  but applies CNN-LSTM to driver drowsiness detection.

  CNN extracts spatial features from each frame.
  LSTM models temporal dependencies across frame sequences.

INPUT FORMAT:
  - Each sample is a video clip tensor:
    (seq_len, img_size, img_size, channels) = (16, 48, 48, 1)
  - Labels:
    0 -> Alert
    1 -> Drowsy
"""

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SEED = 42

CONFIG = {
    "seq_len": 16,
    "img_size": 48,
    "channels": 1,
    "n_classes": 2,
    "n_samples": 1200,
    "test_size": 0.20,
    "val_size": 0.20,
    "cnn_filters_1": 32,
    "cnn_filters_2": 64,
    "cnn_filters_3": 128,
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dense_units": 64,
    "dropout_cnn": 0.2,
    "dropout_lstm": 0.3,
    "dropout_dense": 0.4,
    "l2_reg": 1e-4,
    "label_smoothing": 0.05,
    "use_train_augmentation": True,
    "aug_noise_std": 0.03,
    "aug_brightness_range": 0.20,
    "learning_rate": 1e-3,
    "batch_size": 16,
    "epochs": 15,
    "patience": 4,
}

CLASS_LABELS = {
    0: "Alert",
    1: "Drowsy",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def draw_rect(frame: np.ndarray, top: int, left: int, height: int, width: int, value: float) -> None:
    h, w = frame.shape
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, top + max(1, height))
    right = min(w, left + max(1, width))
    if bottom > top and right > left:
        frame[top:bottom, left:right] = value


def draw_eye(frame: np.ndarray, center_y: int, center_x: int, openness: float) -> None:
    eye_half_w = 5
    eye_h = max(1, int(1 + openness * 4))
    draw_rect(frame, center_y - eye_h // 2, center_x - eye_half_w, eye_h, eye_half_w * 2, 0.95)

    if openness > 0.15:
        pupil_h = max(1, eye_h // 2)
        draw_rect(frame, center_y - pupil_h // 2, center_x - 1, pupil_h, 2, 0.10)


def make_face_frame(img_size: int, openness: float, mouth_open: float, head_shift: float) -> np.ndarray:
    frame = np.full((img_size, img_size), 0.08, dtype=np.float32)

    yy, xx = np.ogrid[:img_size, :img_size]
    center_y = img_size // 2 + int(head_shift)
    center_x = img_size // 2
    radius = img_size // 3
    face_mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius ** 2
    frame[face_mask] = 0.30

    eye_y = center_y - img_size // 10
    eye_dx = img_size // 8
    draw_eye(frame, eye_y, center_x - eye_dx, openness)
    draw_eye(frame, eye_y, center_x + eye_dx, openness)

    mouth_w = img_size // 6
    mouth_h = max(1, int(2 + mouth_open * 4))
    draw_rect(frame, center_y + img_size // 8, center_x - mouth_w // 2, mouth_h, mouth_w, 0.82)

    draw_rect(frame, center_y - 1, center_x, 3, 1, 0.55)

    noise = np.random.normal(0.0, 0.02, size=frame.shape).astype(np.float32)
    frame = np.clip(frame + noise, 0.0, 1.0)

    return frame[..., None]


def simulate_temporal_patterns(label: int, seq_len: int):
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)

    if label == 0:
        openness = 0.82 + 0.08 * np.sin(2 * np.pi * 2.5 * t + np.random.uniform(0, np.pi))
        blink_count = np.random.randint(0, 3)
        for _ in range(blink_count):
            start = np.random.randint(0, max(1, seq_len - 1))
            openness[start:start + 1] = 0.05

        mouth_open = 0.15 + 0.05 * np.sin(2 * np.pi * 1.5 * t)
        head_shift = 1.5 * np.sin(2 * np.pi * 0.8 * t)

    else:
        openness = 0.35 + 0.12 * np.sin(2 * np.pi * 1.0 * t + np.random.uniform(0, np.pi))
        closure_count = np.random.randint(1, 3)
        for _ in range(closure_count):
            length = np.random.randint(3, max(4, seq_len // 2 + 1))
            start = np.random.randint(0, seq_len - length + 1)
            openness[start:start + length] = 0.01

        mouth_open = 0.28 + 0.30 * np.maximum(0.0, np.sin(2 * np.pi * 1.2 * t))
        head_shift = 2.5 * np.sin(2 * np.pi * 0.6 * t) + np.linspace(0.0, 2.0, seq_len)

    openness = np.clip(openness, 0.0, 1.0)
    mouth_open = np.clip(mouth_open, 0.0, 1.0)
    return openness, mouth_open, head_shift


def generate_synthetic_drowsiness_data(
    n_samples: int,
    n_classes: int,
    seq_len: int,
    img_size: int,
    channels: int,
    seed: int = 42,
):
    if channels != 1:
        raise ValueError("Synthetic generator currently supports channels=1 only.")

    np.random.seed(seed)
    X = np.zeros((n_samples, seq_len, img_size, img_size, channels), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int32)

    for i in range(n_samples):
        label = i % n_classes
        y[i] = label

        openness, mouth_open, head_shift = simulate_temporal_patterns(label, seq_len)

        for t_idx in range(seq_len):
            frame = make_face_frame(
                img_size=img_size,
                openness=float(openness[t_idx]),
                mouth_open=float(mouth_open[t_idx]),
                head_shift=float(head_shift[t_idx]),
            )
            brightness = np.random.uniform(0.9, 1.1)
            X[i, t_idx] = np.clip(frame * brightness, 0.0, 1.0)

    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def normalize_splits(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    if X_train.max() > 1.5:
        X_train /= 255.0
        X_val /= 255.0
        X_test /= 255.0

    mean = X_train.mean(axis=(0, 1, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 1, 2, 3), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_val, X_test, mean, std


def to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    return tf.keras.utils.to_categorical(y, num_classes=n_classes).astype(np.float32)


def augment_sequence(x: tf.Tensor, y: tf.Tensor, config: dict):
    if config["aug_brightness_range"] > 0:
        factor = tf.random.uniform(
            shape=(),
            minval=1.0 - float(config["aug_brightness_range"]),
            maxval=1.0 + float(config["aug_brightness_range"]),
            dtype=x.dtype,
        )
        x = x * factor

    if config["aug_noise_std"] > 0:
        noise = tf.random.normal(shape=tf.shape(x), stddev=float(config["aug_noise_std"]), dtype=x.dtype)
        x = x + noise

    shift = tf.random.uniform(shape=(), minval=-2, maxval=3, dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=2)
    x = tf.clip_by_value(x, -4.0, 4.0)
    return x, y


def make_datasets(
    X_train: np.ndarray,
    y_train_oh: np.ndarray,
    X_val: np.ndarray,
    y_val_oh: np.ndarray,
    X_test: np.ndarray,
    y_test_oh: np.ndarray,
    config: dict,
):
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
    train_ds = train_ds.shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)

    if config["use_train_augmentation"]:
        train_ds = train_ds.map(
            lambda x, y: augment_sequence(x, y, config),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = train_ds.batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_oh))
    val_ds = val_ds.batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test_oh))
    test_ds = test_ds.batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    n_classes: int,
    groups: np.ndarray | None = None,
):
    split_info = {
        "strategy": "stratified_random",
        "group_leakage_count": 0,
    }

    if groups is None or len(np.unique(groups)) < 3:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=SEED,
            stratify=y,
        )

        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_ratio,
            random_state=SEED,
            stratify=y_train_val,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test, split_info

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    idx_train_val, idx_test = next(gss_test.split(X, y, groups=groups))

    X_train_val = X[idx_train_val]
    y_train_val = y[idx_train_val]
    g_train_val = groups[idx_train_val]

    X_test = X[idx_test]
    y_test = y[idx_test]
    g_test = groups[idx_test]

    val_ratio = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED + 1)
    idx_train, idx_val = next(gss_val.split(X_train_val, y_train_val, groups=g_train_val))

    X_train = X_train_val[idx_train]
    y_train = y_train_val[idx_train]
    g_train = g_train_val[idx_train]

    X_val = X_train_val[idx_val]
    y_val = y_train_val[idx_val]
    g_val = g_train_val[idx_val]

    class_ok = all(len(np.unique(split_y)) == n_classes for split_y in (y_train, y_val, y_test))
    if not class_ok:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=SEED,
            stratify=y,
        )
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_ratio,
            random_state=SEED,
            stratify=y_train_val,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test, split_info

    leakage_train_test = len(np.intersect1d(g_train, g_test))
    leakage_val_test = len(np.intersect1d(g_val, g_test))
    leakage_train_val = len(np.intersect1d(g_train, g_val))

    split_info = {
        "strategy": "group_shuffle_split",
        "train_groups": int(len(np.unique(g_train))),
        "val_groups": int(len(np.unique(g_val))),
        "test_groups": int(len(np.unique(g_test))),
        "group_leakage_count": int(leakage_train_test + leakage_val_test + leakage_train_val),
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, split_info


def build_cnn_lstm_model(config: dict) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(config["l2_reg"])

    frame_encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(config["img_size"], config["img_size"], config["channels"])),
            tf.keras.layers.Conv2D(
                config["cnn_filters_1"],
                3,
                padding="same",
                activation="relu",
                kernel_regularizer=reg,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(
                config["cnn_filters_2"],
                3,
                padding="same",
                activation="relu",
                kernel_regularizer=reg,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(
                config["cnn_filters_3"],
                3,
                padding="same",
                activation="relu",
                kernel_regularizer=reg,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
        ],
        name="frame_encoder",
    )

    inputs = tf.keras.layers.Input(
        shape=(config["seq_len"], config["img_size"], config["img_size"], config["channels"])
    )

    x = tf.keras.layers.TimeDistributed(frame_encoder)(inputs)
    x = tf.keras.layers.Dropout(config["dropout_cnn"])(x)

    x = tf.keras.layers.LSTM(
        config["lstm_units_1"],
        return_sequences=True,
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
    )(x)
    x = tf.keras.layers.Dropout(config["dropout_lstm"])(x)

    x = tf.keras.layers.LSTM(
        config["lstm_units_2"],
        return_sequences=False,
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
    )(x)
    x = tf.keras.layers.Dropout(config["dropout_lstm"])(x)

    x = tf.keras.layers.Dense(config["dense_units"], activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(config["dropout_dense"])(x)

    outputs = tf.keras.layers.Dense(config["n_classes"], activation="softmax", kernel_regularizer=reg)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_driver_drowsiness")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config["label_smoothing"]),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def load_or_generate_data(project_root: Path, config: dict):
    data_path = project_root / "data" / "drowsiness_sequences.npz"
    groups = None

    if data_path.exists():
        data = np.load(data_path)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)

        for key in ("groups", "subject_ids", "driver_ids"):
            if key in data:
                groups = data[key].astype(np.int32).reshape(-1)
                break

        if X.ndim != 5:
            raise ValueError("Expected X with shape (samples, seq_len, img_size, img_size, channels).")

        expected = (
            config["seq_len"],
            config["img_size"],
            config["img_size"],
            config["channels"],
        )
        got = (X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        if got != expected:
            raise ValueError(f"Expected X shape (*, {expected}), got (*, {got}).")

        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("Expected y with shape (samples,) and same length as X.")

        if groups is not None and len(groups) != len(X):
            raise ValueError("Expected groups/subject_ids with shape (samples,) and same length as X.")

        source = "custom_npz"

    else:
        X, y = generate_synthetic_drowsiness_data(
            n_samples=config["n_samples"],
            n_classes=config["n_classes"],
            seq_len=config["seq_len"],
            img_size=config["img_size"],
            channels=config["channels"],
            seed=SEED,
        )
        source = "synthetic"

    return X, y, source, groups


def main():
    project_root = Path(__file__).resolve().parent
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(SEED)

    print("=" * 68)
    print("  CNN-LSTM Driver Drowsiness Detection")
    print("=" * 68)

    X, y, source, groups = load_or_generate_data(project_root, CONFIG)
    n_classes = int(np.max(y)) + 1

    run_config = dict(CONFIG)
    run_config["n_classes"] = n_classes

    class_names = [CLASS_LABELS.get(i, f"Class_{i}") for i in range(n_classes)]

    X_train, X_val, X_test, y_train, y_val, y_test, split_info = split_data(
        X=X,
        y=y,
        test_size=run_config["test_size"],
        val_size=run_config["val_size"],
        n_classes=run_config["n_classes"],
        groups=groups,
    )

    X_train, X_val, X_test, mean, std = normalize_splits(X_train, X_val, X_test)

    y_train_oh = to_one_hot(y_train, run_config["n_classes"])
    y_val_oh = to_one_hot(y_val, run_config["n_classes"])
    y_test_oh = to_one_hot(y_test, run_config["n_classes"])

    train_ds, val_ds, test_ds = make_datasets(
        X_train,
        y_train_oh,
        X_val,
        y_val_oh,
        X_test,
        y_test_oh,
        run_config,
    )

    print(f"Data source      : {source}")
    print(f"Split strategy   : {split_info['strategy']}")
    if "group_leakage_count" in split_info:
        print(f"Group leakage    : {split_info['group_leakage_count']}")
    print(f"Train shape      : {X_train.shape}")
    print(f"Validation shape : {X_val.shape}")
    print(f"Test shape       : {X_test.shape}")
    print(f"Classes          : {class_names}")
    print(f"Normalization    : mean={float(mean.mean()):.4f}, std={float(std.mean()):.4f}")

    model = build_cnn_lstm_model(run_config)
    model.summary()

    best_model_path = artifacts_dir / "best_cnn_lstm_driver_drowsiness.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=run_config["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=run_config["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    train_metrics = model.evaluate(train_ds, verbose=0, return_dict=True)
    val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
    eval_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    if not isinstance(train_metrics, dict) or not isinstance(val_metrics, dict):
        raise TypeError("Expected train/val evaluate(..., return_dict=True) to return dicts.")
    if not isinstance(eval_metrics, dict):
        raise TypeError("Expected evaluate(..., return_dict=True) to return a dict.")

    test_loss = float(eval_metrics.get("loss", 0.0))
    test_acc = float(eval_metrics.get("accuracy", 0.0))
    test_auc = float(eval_metrics.get("auc", 0.0))

    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    train_acc = float(train_metrics.get("accuracy", 0.0))
    val_acc = float(val_metrics.get("accuracy", 0.0))
    overfit_gap = train_acc - val_acc

    report_text = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    model.save(artifacts_dir / "final_cnn_lstm_driver_drowsiness.keras")

    history_path = artifacts_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    results = {
        "project": "CNN-LSTM Driver Drowsiness Detection",
        "data_source": source,
        "config": run_config,
        "class_names": class_names,
        "metrics": {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_auc": test_auc,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "overfit_gap_acc": float(overfit_gap),
        },
        "split_info": split_info,
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }

    results_path = artifacts_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 68)
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc:.4f}")
    print(f"Test AUC  : {test_auc:.4f}")
    print(f"Train Acc : {train_acc:.4f}")
    print(f"Val Acc   : {val_acc:.4f}")
    print(f"Gap (Tr-V): {overfit_gap:.4f}")
    print("Classification Report:")
    print(report_text)
    print(f"Saved best model : {best_model_path}")
    print(f"Saved history    : {history_path}")
    print(f"Saved results    : {results_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
