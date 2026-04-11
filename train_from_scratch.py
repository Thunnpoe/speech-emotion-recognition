import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

UI6_ORDER = ["fear", "angry", "neutral", "happy", "sad", "surprise"]
RAW_EMOTION_TO_UI6 = {
    "fear": "fear",
    "fearful": "fear",
    "angry": "angry",
    "disgust": "angry",
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}


def normalize_emotion_label(raw_label: str) -> str:
    """Reduce raw dataset emotion labels to the six-class UI taxonomy."""
    token = str(raw_label).strip().lower().split("_")[0]
    return RAW_EMOTION_TO_UI6.get(token, token)


def build_file_index(dataset_roots: list[str] | None) -> tuple[dict[str, str], set[str]]:
    if not dataset_roots:
        return {}, set()
    grouped: dict[str, list[str]] = {}
    for dataset_root in dataset_roots:
        if not dataset_root or not os.path.isdir(dataset_root):
            continue
        for wav_path in Path(dataset_root).rglob("*.wav"):
            grouped.setdefault(wav_path.name, []).append(str(wav_path))

    index: dict[str, str] = {}
    ambiguous: set[str] = set()
    for name, paths in grouped.items():
        if len(paths) == 1:
            index[name] = paths[0]
        else:
            ambiguous.add(name)
    return index, ambiguous


def get_mfccs(audio_path: str, limit: int) -> np.ndarray:
    y, sr = librosa.load(audio_path)
    a = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    else:
        mfccs = np.zeros((a.shape[0], limit), dtype=np.float32)
        mfccs[:, : a.shape[1]] = a
    return mfccs


def build_cnn(input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def resolve_audio_path(
    raw_path: str,
    dataset_roots: list[str] | None,
    path_prefix_from: str | None,
    path_prefix_to: str | None,
    file_index: dict[str, str] | None = None,
    source: str | None = None,
) -> str | None:
    raw_path = str(raw_path)
    if os.path.isfile(raw_path):
        return raw_path
    if path_prefix_from and path_prefix_to and raw_path.startswith(path_prefix_from):
        replaced = raw_path.replace(path_prefix_from, path_prefix_to, 1)
        if os.path.isfile(replaced):
            return replaced
    
    filename = os.path.basename(raw_path)
    
    # If we have source info, search recursively in the matching dataset
    if source and dataset_roots:
        source_lower = source.lower()
        for root in dataset_roots:
            root_lower = root.lower()
            # Match source to dataset root
            if ('tess' in source_lower and 'tess' in root_lower) or \
               ('crema' in source_lower and 'crema' in root_lower) or \
               ('savee' in source_lower and 'savee' in root_lower) or \
               ('ravdess' in source_lower and 'ravdess' in root_lower):
                # Recursive search in this dataset
                for dirpath, dirnames, filenames in os.walk(root):
                    if filename in filenames:
                        return os.path.join(dirpath, filename)
    
    # Fallback: search all datasets recursively
    if dataset_roots:
        for root in dataset_roots:
            for dirpath, dirnames, filenames in os.walk(root):
                if filename in filenames:
                    return os.path.join(dirpath, filename)
    
    # Last resort: use file index if available
    if file_index and filename in file_index:
        return file_index[filename]
    
    return None


def build_dataset(
    df: pd.DataFrame,
    label_col: str,
    max_len: int,
    dataset_roots: list[str] | None,
    path_prefix_from: str | None,
    path_prefix_to: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    file_index, ambiguous_names = build_file_index(dataset_roots)
    total_rows = len(df)
    resolved_rows = 0
    missing_rows = 0
    mfcc_error_rows = 0

    for _, row in df.iterrows():
        raw_path = str(row["path"])
        source = row.get("source", None)  # Get source information if available
        path = resolve_audio_path(
            raw_path=raw_path,
            dataset_roots=dataset_roots,
            path_prefix_from=path_prefix_from,
            path_prefix_to=path_prefix_to,
            file_index=file_index,
            source=source,
        )
        if path is None:
            missing_rows += 1
            continue
        try:
            mfcc = get_mfccs(path, max_len)
            features.append(mfcc)
            labels.append(str(row[label_col]))
            resolved_rows += 1
        except Exception:
            mfcc_error_rows += 1
            continue

    if not features:
        raise RuntimeError("No valid audio files were loaded for training.")

    print(
        f"Loaded {resolved_rows}/{total_rows} rows for '{label_col}'. "
        f"Missing paths: {missing_rows}, MFCC errors: {mfcc_error_rows}."
    )
    if ambiguous_names:
        print(
            f"Warning: {len(ambiguous_names)} duplicate .wav basenames were detected across dataset roots. "
            "Ambiguous names are ignored by fallback matching."
        )

    x = np.array(features, dtype=np.float32)
    x = x[..., np.newaxis]
    y = np.array(labels)
    return x, y


def create_balanced_dataset(x, y, max_samples_per_class=None):
    """Create a balanced dataset by oversampling minority classes."""
    from sklearn.utils import resample
    
    unique_classes = np.unique(y)
    balanced_x, balanced_y = [], []
    
    if max_samples_per_class is None:
        # Use the size of the largest class
        class_counts = {cls: np.sum(y == cls) for cls in unique_classes}
        max_samples_per_class = max(class_counts.values())
    
    print(f"Balancing dataset to {max_samples_per_class} samples per class")
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        cls_x = x[cls_indices]
        cls_y = y[cls_indices]
        
        if len(cls_indices) < max_samples_per_class:
            # Oversample
            cls_x_resampled, cls_y_resampled = resample(
                cls_x, cls_y, 
                n_samples=max_samples_per_class, 
                random_state=42,
                replace=True
            )
            print(f"  {cls}: {len(cls_indices)} -> {max_samples_per_class} (oversampled)")
        else:
            # Use original samples
            cls_x_resampled, cls_y_resampled = cls_x[:max_samples_per_class], cls_y[:max_samples_per_class]
            print(f"  {cls}: {len(cls_indices)} -> {max_samples_per_class} (undersampled)")
        
        balanced_x.append(cls_x_resampled)
        balanced_y.append(cls_y_resampled)
    
    return np.concatenate(balanced_x), np.concatenate(balanced_y)


def train_one(
    x: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    output_model: str,
    output_labels: str,
    epochs: int,
    batch_size: int,
    seed: int,
    resume: bool,
) -> None:
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    y_encoded = np.array([class_to_idx[label] for label in y], dtype=np.int32)
    y_cat = to_categorical(y_encoded, num_classes=len(class_names))

    # Balance the dataset for emotion recognition (but not for gender)
    if len(class_names) > 3:  # Emotion models have 6 classes, gender has 2
        print("Creating balanced dataset for emotion training...")
        x_balanced, y_balanced_encoded = create_balanced_dataset(x, y_encoded)
        y_balanced_cat = to_categorical(y_balanced_encoded, num_classes=len(class_names))
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_balanced, y_balanced_cat,
            test_size=0.2, random_state=seed, stratify=y_balanced_encoded
        )
        y_train_idx = np.argmax(y_train, axis=1)
        y_val_idx = np.argmax(y_val, axis=1)
    else:
        # Original split for gender model
        x_train, x_val, y_train, y_val = train_test_split(
            x, y_cat, test_size=0.2, random_state=seed, stratify=y_encoded
        )
        y_train_idx = np.argmax(y_train, axis=1)
        y_val_idx = np.argmax(y_val, axis=1)

    if resume and os.path.isfile(output_model):
        model = load_model(output_model)
    else:
        model = build_cnn(input_shape=x.shape[1:], num_classes=len(class_names))

    present_classes = np.unique(y_train_idx)
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train_idx,
    )
    class_weight = {i: 1.0 for i in range(len(class_names))}
    for cls_idx, w in zip(present_classes, class_weights_arr):
        class_weight[int(cls_idx)] = float(w)

    print(f"Training emotion model with {len(class_names)} classes: {class_names}")
    print(f"Class weights: {class_weight}")
    print(f"Training data shape: {x_train.shape}, Validation data shape: {x_val.shape}")
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(output_model, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    y_pred = model.predict(x_val, verbose=0).argmax(axis=1)
    print("\nValidation classification report:")
    print(
        classification_report(
            y_val_idx,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )
    print("Validation confusion matrix:")
    print(confusion_matrix(y_val_idx, y_pred, labels=np.arange(len(class_names))))

    Path(output_labels).write_text("\n".join(class_names), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SER models from scratch.")
    parser.add_argument("--csv", default="df_audio.csv", help="Path to training CSV file")
    parser.add_argument(
        "--dataset-root",
        nargs="+",
        default=None,
        help="One or more folders with audio files if CSV paths are not directly valid",
    )
    parser.add_argument(
        "--path-prefix-from",
        default=None,
        help="Optional old path prefix to replace (example: /gdrive/MyDrive/Projects/ITCproject/data3)",
    )
    parser.add_argument(
        "--path-prefix-to",
        default=None,
        help="Optional local replacement prefix for --path-prefix-from",
    )
    parser.add_argument("--max-len", type=int, default=216, help="MFCC time-axis length")
    parser.add_argument(
        "--emotion-label-col",
        default="emotion_label",
        help="CSV column used for emotion labels (default: emotion_label)",
    )
    parser.add_argument(
        "--emotion-label-mode",
        choices=["ui6", "raw"],
        default="ui6",
        help="ui6 maps raw labels to fear/angry/neutral/happy/sad/surprise; raw uses labels as-is",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from existing self-trained model files if they exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv)
    if "path" not in df.columns:
        raise ValueError("CSV must contain a 'path' column.")
    if args.emotion_label_col not in df.columns:
        raise ValueError(f"CSV must contain '{args.emotion_label_col}' column.")
    if "actors" not in df.columns:
        raise ValueError("CSV must contain 'actors' column.")

    if args.emotion_label_mode == "ui6":
        df["_emotion_train"] = df[args.emotion_label_col].astype(str).map(normalize_emotion_label)
        emotion_classes = [name for name in UI6_ORDER if name in set(df["_emotion_train"].unique())]
    else:
        df["_emotion_train"] = df[args.emotion_label_col].astype(str).str.strip().str.lower()
        emotion_classes = sorted(df["_emotion_train"].unique().tolist())

    if not emotion_classes:
        raise RuntimeError("No emotion classes found after label preprocessing.")
    df = df[df["_emotion_train"].isin(emotion_classes)].copy()

    gender_classes = sorted(df["actors"].astype(str).unique().tolist())

    x_emotion, y_emotion = build_dataset(
        df=df,
        label_col="_emotion_train",
        max_len=args.max_len,
        dataset_roots=args.dataset_root,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
    )
    train_one(
        x=x_emotion,
        y=y_emotion,
        class_names=emotion_classes,
        output_model="model3_self_trained.h5",
        output_labels="model3_self_trained.labels.txt",
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        resume=args.resume,
    )

    x_gender, y_gender = build_dataset(
        df=df,
        label_col="actors",
        max_len=args.max_len,
        dataset_roots=args.dataset_root,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
    )
    train_one(
        x=x_gender,
        y=y_gender,
        class_names=gender_classes,
        output_model="model_mw_self_trained.h5",
        output_labels="model_mw_self_trained.labels.txt",
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        resume=args.resume,
    )

    print("Training complete.")
    print("Created: model3_self_trained.h5, model_mw_self_trained.h5")


if __name__ == "__main__":
    main()
