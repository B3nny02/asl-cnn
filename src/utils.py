"""
utils.py
Data utilities for ASL-CNN
Francesco Benincasa – MIT Licence
"""
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def locate_dataset_dirs(base_dir):
    """
    Returns absolute paths to the training and test folders.
    Works around possible kagglehub extraction quirks.
    """
    train_candidates = [
        os.path.join(base_dir, 'asl_alphabet_train', 'asl_alphabet_train'),
        os.path.join(base_dir, 'asl_alphabet_train'),
        base_dir
    ]
    for d in train_candidates:
        if os.path.isdir(d) and any(os.path.isdir(os.path.join(d, sub))
                                    for sub in os.listdir(d)
                                    if not sub.startswith('.')):
            train_dir = d
            break
    else:
        raise FileNotFoundError("Could not locate training folder.")

    test_candidates = [
        os.path.join(base_dir, 'asl_alphabet_test', 'asl_alphabet_test'),
        os.path.join(base_dir, 'asl_alphabet_test')
    ]
    for d in test_candidates:
        if os.path.isdir(d):
            test_dir = d
            break
    else:
        test_dir = None   # will use validation split instead
    return train_dir, test_dir


def create_generators(train_dir,
                      img_size=(200, 200),
                      batch_size=32,
                      val_split=0.2):
    """
    Returns (train_generator, val_generator, class_labels)
    """
    # Detect classes automatically
    class_labels = sorted([d for d in os.listdir(train_dir)
                           if os.path.isdir(os.path.join(train_dir, d))])

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        validation_split=val_split
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_labels,
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_labels,
        subset='validation'
    )
    return train_gen, val_gen, class_labels
