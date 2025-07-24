#!/usr/bin/env python3
"""
train.py
CLI entry point for training the ASL-CNN
Francesco Benincasa – MIT Licence
"""
import argparse
import os
import kagglehub
from model import build_model
from utils import locate_dataset_dirs, create_generators
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def main():
    parser = argparse.ArgumentParser(description="Train ASL-CNN")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--kernel", type=int, default=3, choices=[3, 5])
    parser.add_argument("--outdir", default="../checkpoints")
    args = parser.parse_args()

    # 1. Locate dataset
    dataset_base = kagglehub.dataset_download("grassknoted/asl-alphabet")
    train_dir, _ = locate_dataset_dirs(dataset_base)

    # 2. Data loaders
    train_gen, val_gen, classes = create_generators(train_dir, batch_size=args.batch)

    # 3. Model
    model = build_model(
        num_classes=len(classes),
        dropout_rate=args.dropout,
        kernel_size=args.kernel
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Callbacks
    os.makedirs(args.outdir, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    # 5. Train
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Save final weights
    model.save(os.path.join(args.outdir, "final.h5"))
    print("Training complete – weights saved to", args.outdir)

if __name__ == "__main__":
    main()
