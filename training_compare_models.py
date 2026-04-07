import os
import csv
import argparse
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Conv2DTranspose,
    Concatenate,
    BatchNormalization,
    Activation,
    Add,
)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from skimage import io
from unet import UNet


def pick_rgb_candidate(img_dir):
    rgb_files = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith('.bmp') and 'rgb' in f.lower()
    )
    if not rgb_files:
        return None
    return os.path.join(img_dir, rgb_files[0])


def load_dataset(img_dir, mask_dir, texture_dir, rgb_path=None):
    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith('.bmp'))
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith('.bmp'))
    tex_files = sorted(f for f in os.listdir(texture_dir) if f.lower().endswith('.bmp'))

    if len(mask_files) == 0 or len(tex_files) == 0 or len(img_files) == 0:
        raise RuntimeError('Пустой набор данных: проверьте img/mask/texture')

    # Prefer non-rgb files as base image list when both gray and rgb variants exist.
    base_img_files = [f for f in img_files if 'rgb' not in f.lower()]
    if not base_img_files:
        base_img_files = img_files

    sample_count = min(len(mask_files), len(tex_files), len(base_img_files))

    rgb_override = None
    if rgb_path and os.path.isfile(rgb_path):
        rgb_override = io.imread(rgb_path)
    elif rgb_path is None:
        auto_rgb = pick_rgb_candidate(img_dir)
        if auto_rgb:
            rgb_override = io.imread(auto_rgb)

    images = []
    masks = []

    for idx in range(sample_count):
        base_img = io.imread(os.path.join(img_dir, base_img_files[idx]))
        mask = io.imread(os.path.join(mask_dir, mask_files[idx]))
        texture = io.imread(os.path.join(texture_dir, tex_files[idx]))

        if mask.ndim == 3:
            mask = mask[..., 0]

        if texture.ndim == 3:
            texture = texture[..., 0]

        # Use an external rgb override only in one-image setup to avoid copying one rgb frame
        # to unrelated samples when dataset has many images.
        if sample_count == 1 and rgb_override is not None and rgb_override.shape[:2] == mask.shape[:2]:
            rgb = rgb_override
        elif base_img.ndim == 3 and base_img.shape[-1] >= 3:
            rgb = base_img[..., :3]
        else:
            gray = base_img[..., 0] if base_img.ndim == 3 else base_img
            rgb = np.stack([gray, gray, gray], axis=-1)

        merged = np.concatenate(
            [rgb.astype(np.float32), texture[..., np.newaxis].astype(np.float32)], axis=-1
        )

        images.append(merged)
        masks.append(mask)

    return images, masks


def crop_dataset(images, masks, tile_size, stride):
    out_images = []
    out_masks = []

    for image, mask in zip(images, masks):
        height, width = image.shape[:2]

        y_positions = list(range(0, max(height - tile_size, 0) + 1, stride))
        x_positions = list(range(0, max(width - tile_size, 0) + 1, stride))

        if len(y_positions) == 0 or y_positions[-1] != height - tile_size:
            y_positions.append(height - tile_size)
        if len(x_positions) == 0 or x_positions[-1] != width - tile_size:
            x_positions.append(width - tile_size)

        for y_pos in y_positions:
            for x_pos in x_positions:
                out_images.append(image[y_pos:y_pos + tile_size, x_pos:x_pos + tile_size])
                out_masks.append(mask[y_pos:y_pos + tile_size, x_pos:x_pos + tile_size])

    return np.array(out_images), np.array(out_masks)


def masks_to_onehot(mask_set):
    object_mask = (mask_set == 255).astype(np.float32)
    background = 1.0 - object_mask
    return np.stack([background, object_mask], axis=-1)


def trim_to_batch(x_data, y_data, batch_size):
    remainder = len(x_data) % batch_size
    if remainder == 0:
        return x_data, y_data
    keep = len(x_data) - remainder
    return x_data[:keep], y_data[:keep]


def make_blend_window(tile_size, eps=1e-3):
    one_dim = np.hanning(tile_size).astype(np.float32)
    one_dim = np.maximum(one_dim, eps)
    return np.outer(one_dim, one_dim).astype(np.float32)


def predict_full_probability(model, merged_image, tile_size, batch_size, stride):
    height, width = merged_image.shape[:2]

    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))

    if len(ys) == 0 or ys[-1] != height - tile_size:
        ys.append(height - tile_size)
    if len(xs) == 0 or xs[-1] != width - tile_size:
        xs.append(width - tile_size)

    patches = []
    patch_pos = []

    for y_pos in ys:
        for x_pos in xs:
            patch = merged_image[y_pos:y_pos + tile_size, x_pos:x_pos + tile_size]
            patches.append((patch / 255.0).astype(np.float32))
            patch_pos.append((y_pos, x_pos))

    patches = np.array(patches)
    preds = model.predict(patches, batch_size=batch_size, verbose=0)
    blend = make_blend_window(tile_size)

    prob_sum = np.zeros((height, width), dtype=np.float32)
    prob_count = np.zeros((height, width), dtype=np.float32)

    for idx, (y_pos, x_pos) in enumerate(patch_pos):
        prob_sum[y_pos:y_pos + tile_size, x_pos:x_pos + tile_size] += preds[idx, :, :, 1] * blend
        prob_count[y_pos:y_pos + tile_size, x_pos:x_pos + tile_size] += blend

    return prob_sum / np.maximum(prob_count, 1e-8)


def pick_threshold_by_val_f1(y_true, y_prob, low=0.2, high=0.8, points=61):
    flat_true = y_true.astype(np.uint8).ravel()
    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.linspace(low, high, points):
        flat_pred = (y_prob >= thr).astype(np.uint8).ravel()
        score = f1_score(flat_true, flat_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)

    return best_thr, best_f1


def conv_bn_relu(x, filters, kernel=3):
    x = Conv2D(filters, kernel, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


def residual_block(x, filters):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = conv_bn_relu(x, filters)
    y = Conv2D(filters, 3, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)

    out = Add()([shortcut, y])
    return Activation('relu')(out)


def csp_block(x, filters):
    shortcut = conv_bn_relu(x, filters, kernel=1)
    y = conv_bn_relu(x, filters)
    y = conv_bn_relu(y, filters)
    y = Concatenate()([shortcut, y])
    return conv_bn_relu(y, filters, kernel=1)


def build_yolo_seg_lite(input_shape, out_ch=2):
    inp = Input(shape=input_shape)

    s1 = conv_bn_relu(inp, 32)
    x = Conv2D(64, 3, strides=2, padding='same', use_bias=False)(s1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    s2 = csp_block(x, 64)
    x = Conv2D(128, 3, strides=2, padding='same', use_bias=False)(s2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = csp_block(x, 128)
    x = conv_bn_relu(x, 128)

    x = Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([x, s2])
    x = conv_bn_relu(x, 64)

    x = Conv2DTranspose(32, 3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([x, s1])
    x = conv_bn_relu(x, 32)

    out = Conv2D(out_ch, 1, activation='softmax')(x)
    return Model(inp, out, name='YOLOSegLite')


def build_resunet(input_shape, out_ch=2):
    inp = Input(shape=input_shape)

    e1 = residual_block(inp, 32)
    p1 = MaxPooling2D()(e1)

    e2 = residual_block(p1, 64)
    p2 = MaxPooling2D()(e2)

    b = residual_block(p2, 128)

    u2 = UpSampling2D()(b)
    u2 = Concatenate()([u2, e2])
    u2 = residual_block(u2, 64)

    u1 = UpSampling2D()(u2)
    u1 = Concatenate()([u1, e1])
    u1 = residual_block(u1, 32)

    out = Conv2D(out_ch, 1, activation='softmax')(u1)
    return Model(inp, out, name='ResUNet')


def build_segnet_lite(input_shape, out_ch=2):
    inp = Input(shape=input_shape)

    e1 = conv_bn_relu(inp, 32)
    p1 = MaxPooling2D()(e1)

    e2 = conv_bn_relu(p1, 64)
    p2 = MaxPooling2D()(e2)

    b = conv_bn_relu(p2, 128)

    d2 = UpSampling2D()(b)
    d2 = conv_bn_relu(d2, 64)

    d1 = UpSampling2D()(d2)
    d1 = conv_bn_relu(d1, 32)

    out = Conv2D(out_ch, 1, activation='softmax')(d1)
    return Model(inp, out, name='SegNetLite')


def build_unet_model(input_shape, out_ch=2):
    return UNet(input_shape, batchnorm=True, dropout=0.0, out_ch=out_ch)


def format_float(value):
    return f'{value:.6f}'


def write_results_table(results, out_dir):
    csv_path = os.path.join(out_dir, 'model_comparison.csv')
    md_path = os.path.join(out_dir, 'model_comparison.md')

    fields = [
        'model',
        'train_f1',
        'val_f1_best_thr',
        'test_f1',
        'full_macro_f1',
        'threshold',
        'epochs_ran',
        'params',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    lines = [
        '| Model | Train F1 | Val F1@best_thr | Test F1 | Full macro F1 | Threshold | Epochs | Params |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in results:
        lines.append(
            '| {model} | {train_f1} | {val_f1_best_thr} | {test_f1} | {full_macro_f1} | {threshold} | {epochs_ran} | {params} |'.format(
                **row
            )
        )

    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write('\n'.join(lines) + '\n')


def write_run_summary(results, args, out_dir):
    if not results:
        return

    best = max(results, key=lambda row: float(row['test_f1']))
    summary_path = os.path.join(out_dir, 'run_summary.md')

    lines = [
        '# Run Summary',
        '',
        '## Configuration',
        f'- img: `{args.img}`',
        f'- mask: `{args.mask}`',
        f'- texture: `{args.texture}`',
        f'- rgb_image: `{args.rgb_image}`',
        f'- tile: `{args.tile}`',
        f'- stride: `{args.stride}`',
        f'- infer_stride: `{args.infer_stride}`',
        f'- batch: `{args.batch}`',
        f'- epochs: `{args.epochs}`',
        f'- learning_rate: `{args.learning_rate}`',
        f'- patience: `{args.patience}`',
        f'- seed: `{args.seed}`',
        '',
        '## Best Model',
        f"- model: `{best['model']}`",
        f"- test_f1: `{best['test_f1']}`",
        f"- full_macro_f1: `{best['full_macro_f1']}`",
        f"- threshold: `{best['threshold']}`",
        '',
        '## Ranking (by test_f1)',
    ]

    for idx, row in enumerate(sorted(results, key=lambda r: float(r['test_f1']), reverse=True), start=1):
        lines.append(
            f"{idx}. `{row['model']}`: test_f1={row['test_f1']}, full_macro_f1={row['full_macro_f1']}, epochs={row['epochs_ran']}"
        )

    lines.extend([
        '',
        'See also: `model_comparison.md` and per-model `metrics.txt` files.',
    ])

    with open(summary_path, 'w', encoding='utf-8') as out:
        out.write('\n'.join(lines) + '\n')


def train_one_model(model_name, model, data, args):
    x_train, y_train, x_val, y_val, x_test, y_test, full_image, full_mask = data

    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    model_dir = os.path.join(args.out, model_name)
    os.makedirs(model_dir, exist_ok=True)

    checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'weights.weights.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=args.patience,
        verbose=1,
        mode='min',
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=args.batch,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint, early_stopping],
        shuffle=True,
        verbose=1,
    )

    train_pred = model.predict(x_train, batch_size=args.batch, verbose=0)
    val_pred = model.predict(x_val, batch_size=args.batch, verbose=0)
    test_pred = model.predict(x_test, batch_size=args.batch, verbose=0)

    threshold, best_val_f1 = pick_threshold_by_val_f1(y_val[:, :, :, 1], val_pred[:, :, :, 1])

    train_f1 = f1_score(
        y_train[:, :, :, 1].astype(np.uint8).ravel(),
        (train_pred[:, :, :, 1] >= threshold).astype(np.uint8).ravel(),
        zero_division=0,
    )
    test_f1 = f1_score(
        y_test[:, :, :, 1].astype(np.uint8).ravel(),
        (test_pred[:, :, :, 1] >= threshold).astype(np.uint8).ravel(),
        zero_division=0,
    )

    full_prob = predict_full_probability(
        model,
        full_image,
        tile_size=args.tile,
        batch_size=args.batch,
        stride=args.infer_stride,
    )
    full_pred = (full_prob >= threshold).astype(np.uint8)
    full_true = (full_mask == 255).astype(np.uint8)
    full_macro_f1 = f1_score(
        full_true.ravel(),
        full_pred.ravel(),
        average='macro',
        zero_division=0,
    )

    full_cls_report = classification_report(
        full_true.ravel(),
        full_pred.ravel(),
        target_names=['Background', 'Object'],
        digits=2,
        zero_division=0,
    )

    io.imsave(os.path.join(model_dir, 'full_prediction.png'), full_pred.astype(np.uint8) * 255)

    summary = {
        'model': model_name,
        'train_f1': format_float(train_f1),
        'val_f1_best_thr': format_float(best_val_f1),
        'test_f1': format_float(test_f1),
        'full_macro_f1': format_float(full_macro_f1),
        'threshold': format_float(threshold),
        'epochs_ran': str(len(history.history.get('loss', []))),
        'params': str(model.count_params()),
    }

    with open(os.path.join(model_dir, 'metrics.txt'), 'w', encoding='utf-8') as out:
        for key, value in summary.items():
            out.write(f'{key}: {value}\n')

    with open(os.path.join(model_dir, 'report.txt'), 'w', encoding='utf-8') as out:
        out.write(full_cls_report + '\n')

    return summary


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = script_dir

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img',
        default=os.path.join(repo_root, 'dataset', 'img'),
        help='Папка с исходными снимками',
    )
    parser.add_argument(
        '--mask',
        default=os.path.join(repo_root, 'dataset', 'mask'),
        help='Папка с масками',
    )
    parser.add_argument(
        '--texture',
        default=os.path.join(repo_root, 'dataset', 'texture'),
        help='Папка с текстурами',
    )
    parser.add_argument('--rgb-image', default=None, help='Путь к RGB изображению (опционально)')
    parser.add_argument('--tile', type=int, default=256, help='Размер патча')
    parser.add_argument('--stride', type=int, default=128, help='Шаг нарезки train/val/test')
    parser.add_argument('--infer-stride', type=int, default=64, help='Шаг тайлинга полного прогноза')
    parser.add_argument('--batch', type=int, default=2, help='Размер batch')
    parser.add_argument('--epochs', type=int, default=12, help='Максимум эпох на модель')
    parser.add_argument('--patience', type=int, default=3, help='Patience для EarlyStopping')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Скорость обучения')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument(
        '--out',
        default=os.path.join(script_dir, 'results', 'models_compare'),
        help='Папка результатов',
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    images, masks = load_dataset(args.img, args.mask, args.texture, rgb_path=args.rgb_image)
    x_data, y_data = crop_dataset(images, masks, tile_size=args.tile, stride=args.stride)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.15,
        random_state=args.seed,
        shuffle=True,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=args.seed,
        shuffle=True,
    )

    x_train, y_train = trim_to_batch(x_train, y_train, args.batch)
    x_val, y_val = trim_to_batch(x_val, y_val, args.batch)
    x_test, y_test = trim_to_batch(x_test, y_test, args.batch)

    x_train = x_train.astype(np.float32) / 255.0
    x_val = x_val.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train = masks_to_onehot(y_train)
    y_val = masks_to_onehot(y_val)
    y_test = masks_to_onehot(y_test)

    input_shape = (args.tile, args.tile, x_train.shape[-1])

    model_builders = [
        ('UNet', lambda: build_unet_model(input_shape, out_ch=2)),
        ('YOLOSegLite', lambda: build_yolo_seg_lite(input_shape, out_ch=2)),
        ('ResUNet', lambda: build_resunet(input_shape, out_ch=2)),
        ('SegNetLite', lambda: build_segnet_lite(input_shape, out_ch=2)),
    ]

    print('Patch split:', 'train', len(x_train), 'val', len(x_val), 'test', len(x_test))
    print('Input shape:', input_shape)

    common_data = (x_train, y_train, x_val, y_val, x_test, y_test, images[0], masks[0])

    results = []
    for model_name, model_builder in model_builders:
        print('\n=== Training model:', model_name, '===')
        tf.keras.backend.clear_session()
        model = model_builder()
        row = train_one_model(model_name, model, common_data, args)
        results.append(row)

    # Удобнее читать итоговую таблицу, если отсортировать по test F1
    results.sort(key=lambda row: float(row['test_f1']), reverse=True)
    write_results_table(results, args.out)
    write_run_summary(results, args, args.out)

    print('\n=== Final F1 table ===')
    for row in results:
        print(row)
    print('Saved table:', os.path.join(args.out, 'model_comparison.md'))


if __name__ == '__main__':
    main()
