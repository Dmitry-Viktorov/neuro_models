import os
import argparse
import numpy as np
import tensorflow as tf
from skimage import io
from skimage import morphology
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from unet import UNet
import matplotlib.pyplot as plt


def load_dataset(img_dir, mask_dir, texture_dir):
    # читаем только bmp и синхронизируем тройки img/mask/texture
    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith('.bmp'))
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith('.bmp'))
    tex_files = sorted(f for f in os.listdir(texture_dir) if f.lower().endswith('.bmp'))
    count = min(len(img_files), len(mask_files), len(tex_files))

    images, masks = [], []
    for index in range(count):
        img = io.imread(os.path.join(img_dir, img_files[index]))
        mask = io.imread(os.path.join(mask_dir, mask_files[index]))
        texture = io.imread(os.path.join(texture_dir, tex_files[index]))

        if mask.ndim == 3:
            mask = mask[..., 0]

        rows, cols = img.shape
        merged = np.zeros((rows, cols, 2), dtype=np.float32)
        merged[:, :, 0] = img
        merged[:, :, 1] = texture

        images.append(merged)
        masks.append(mask)

    return images, masks


def crop_dataset(input_images, input_masks, tile_size, stride=None):
    if stride is None:
        stride = tile_size

    out_images = []
    out_masks = []
    for image_number in range(len(input_images)):
        cur_img = input_images[image_number]
        cur_mask = input_masks[image_number]
        height, width, _ = cur_img.shape
        y_pos = 0
        y_break = False
        while not y_break:
            if y_pos + tile_size > height:
                y_pos = height - tile_size
                y_break = True
            x_pos = 0
            x_break = False
            while not x_break:
                if x_pos + tile_size > width:
                    x_pos = width - tile_size
                    x_break = True
                out_images.append(cur_img[y_pos:y_pos + tile_size, x_pos:x_pos + tile_size])
                out_masks.append(cur_mask[y_pos:y_pos + tile_size, x_pos:x_pos + tile_size])
                x_pos += stride
            y_pos += stride

    return np.array(out_images), np.array(out_masks)


def norm_mask(mask):
    if mask.ndim == 3:
        mask = mask[..., 0]
    object_mask = (mask == 255).astype(np.float32)
    background_mask = 1.0 - object_mask
    return np.stack([background_mask, object_mask], axis=-1)


def masks_to_onehot(mask_set):
    out = np.zeros((len(mask_set), mask_set.shape[1], mask_set.shape[2], 2), dtype=np.float32)
    for index in range(len(mask_set)):
        out[index] = norm_mask(mask_set[index])
    return out


def trim_to_batch(x_data, y_data, batch_size):
    remainder = len(x_data) % batch_size
    if remainder == 0:
        return x_data, y_data
    keep = len(x_data) - remainder
    return x_data[:keep], y_data[:keep]


def get_positions(height, width, tile_size, stride):
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    if len(ys) == 0 or ys[-1] != height - tile_size:
        ys.append(height - tile_size)
    if len(xs) == 0 or xs[-1] != width - tile_size:
        xs.append(width - tile_size)
    return ys, xs


def make_blend_window(tile_size, eps=1e-3):
    one_dim = np.hanning(tile_size).astype(np.float32)
    one_dim = np.maximum(one_dim, eps)
    return np.outer(one_dim, one_dim).astype(np.float32)


def predict_full_probability(model, merged_image, tile_size, batch_size, stride=None):
    if stride is None:
        stride = tile_size

    height, width, _ = merged_image.shape
    ys, xs = get_positions(height, width, tile_size, stride)

    patches = []
    patch_pos = []
    for y_pos in ys:
        for x_pos in xs:
            patch = merged_image[y_pos:y_pos + tile_size, x_pos:x_pos + tile_size] / 255.0
            patches.append(patch.astype(np.float32))
            patch_pos.append((y_pos, x_pos))

    patches = np.array(patches)
    preds = model.predict(patches, batch_size=batch_size, verbose=0)
    blend = make_blend_window(tile_size)

    prob_sum = np.zeros((height, width), dtype=np.float32)
    prob_count = np.zeros((height, width), dtype=np.float32)
    for index, (y_pos, x_pos) in enumerate(patch_pos):
        prob_sum[y_pos:y_pos + tile_size, x_pos:x_pos + tile_size] += preds[index, :, :, 1] * blend
        prob_count[y_pos:y_pos + tile_size, x_pos:x_pos + tile_size] += blend

    return prob_sum / np.maximum(prob_count, 1e-8)


def predict_full_image(model, merged_image, tile_size, batch_size, threshold, clean_min_size, stride=None):
    mean_prob = predict_full_probability(model, merged_image, tile_size, batch_size, stride)
    binary = (mean_prob >= threshold).astype(np.uint8) * 255
    binary = clean_binary_mask(binary, min_size=clean_min_size)
    return binary


def save_history_plot(history, save_path):
    # графики потери и точность
    train_loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    train_acc = history.history.get('accuracy', history.history.get('acc', []))
    val_acc = history.history.get('val_accuracy', history.history.get('val_acc', []))
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, '-o', label='train_loss')
    plt.plot(epochs, val_loss, '-o', label='val_loss')
    plt.title('Loss: train vs val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    if len(train_acc) == len(train_loss) and len(train_acc) > 0:
        plt.plot(epochs, train_acc, '-o', label='train_accuracy')
    if len(val_acc) == len(train_loss) and len(val_acc) > 0:
        plt.plot(epochs, val_acc, '-o', label='val_accuracy')
    plt.title('Accuracy: train vs val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    if (len(train_acc) > 0) or (len(val_acc) > 0):
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_patch_predictions(preds, out_dir, threshold, clean_min_size, limit=3):
    count = min(limit, len(preds))
    for index in range(count):
        cls = (preds[index, :, :, 1] >= threshold).astype(np.uint8) * 255
        cls = clean_binary_mask(cls, min_size=clean_min_size)
        rgb = np.stack([cls, cls, cls], axis=-1)
        io.imsave(os.path.join(out_dir, f'result{index}.png'), rgb)


def clean_binary_mask(binary, min_size=64):
    # легкая очистка от мелкого шума
    positive = binary > 0
    positive = morphology.remove_small_objects(positive, min_size=min_size, connectivity=2)
    positive = morphology.remove_small_holes(positive, area_threshold=min_size, connectivity=2)
    positive = morphology.binary_closing(positive, morphology.disk(1))
    positive = morphology.binary_opening(positive, morphology.disk(1))
    return positive.astype(np.uint8) * 255


def pick_threshold_by_prevalence(preds, positive_rate):
    # адаптивный порог: подгоняем долю позитивных пикселей
    flat = preds[:, :, :, 1].ravel()
    quantile = float(np.clip(1.0 - positive_rate, 0.0, 1.0))
    return float(np.quantile(flat, quantile))


def pick_threshold_by_val_f1(y_true, y_prob, low=0.2, high=0.8, points=61):
    flat_true = y_true.astype(np.uint8).ravel()
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(low, high, points):
        flat_pred = (y_prob >= thr).astype(np.uint8).ravel()
        cur_f1 = f1_score(flat_true, flat_pred, zero_division=0)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_thr = float(thr)
    return best_thr, best_f1


def pick_threshold_by_full_macro_f1(gt_full, full_prob, low=0.2, high=0.9, points=71):
    best_thr = 0.5
    best_score = -1.0
    for thr in np.linspace(low, high, points):
        pred = (full_prob >= thr).astype(np.uint8).ravel()
        score = f1_score(gt_full, pred, average='macro', zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='dataset/img', help='папка с исходными изображениями')
    parser.add_argument('--mask', default='dataset/mask', help='папка с масками')
    parser.add_argument('--texture', default='dataset/texture', help='папка с текстурами')
    parser.add_argument('--tile', type=int, default=256, help='размер плитки')
    parser.add_argument('--stride', type=int, default=256, help='шаг нарезки')
    parser.add_argument('--infer-stride', type=int, default=128, help='шаг тайлинга при прогнозе полного изображения')
    parser.add_argument('--batch', type=int, default=2, help='размер batch')
    parser.add_argument('--epochs', type=int, default=30, help='число эпох')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--clean-min-size', type=int, default=64, help='минимальный размер артефактов для удаления')
    parser.add_argument('--full-threshold-sweep', action='store_true', help='подобрать порог по macro-F1 на полном изображении')
    parser.add_argument('--full-thr-low', type=float, default=0.2, help='нижняя граница sweep порога')
    parser.add_argument('--full-thr-high', type=float, default=0.9, help='верхняя граница sweep порога')
    parser.add_argument('--full-thr-points', type=int, default=71, help='число точек sweep порога')
    parser.add_argument('--out', default='results', help='папка для результатов')
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    source_images, source_masks = load_dataset(args.img, args.mask, args.texture)
    if len(source_images) == 0:
        raise RuntimeError('В dataset нет данных: проверь папки img/mask/texture')

    img_set, mask_set = crop_dataset(source_images, source_masks, args.tile, args.stride)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        img_set,
        mask_set,
        test_size=0.15,
        random_state=args.seed,
        shuffle=True
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=args.seed,
        shuffle=True
    )

    x_train, y_train = trim_to_batch(x_train, y_train, args.batch)
    if len(x_val) > args.batch:
        x_val, y_val = trim_to_batch(x_val, y_val, args.batch)

    x_train = x_train.astype(np.float32) / 255.0
    x_val = x_val.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train = masks_to_onehot(y_train)
    y_val = masks_to_onehot(y_val)
    y_test = masks_to_onehot(y_test)

    print('train:', len(x_train), 'val:', len(x_val), 'test:', len(x_test))

    # базовый UNet 
    model = UNet((args.tile, args.tile, 2), batchnorm=True, dropout=0, out_ch=2)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    checkpoint = ModelCheckpoint(
        os.path.join(args.out, 'weights.weights.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=args.batch,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint, early_stopping],
        shuffle=True,
        verbose=1
    )

    save_history_plot(history, os.path.join(args.out, 'history.png'))

    train_preds = model.predict(x_train, batch_size=args.batch, verbose=0)
    val_preds = model.predict(x_val, batch_size=args.batch, verbose=0)
    threshold, val_best_f1 = pick_threshold_by_val_f1(y_val[:, :, :, 1], val_preds[:, :, :, 1])
    test_preds = model.predict(x_test, batch_size=args.batch, verbose=0)

    train_cls = (train_preds[:, :, :, 1] >= threshold).astype(np.uint8).ravel()
    train_true = y_train[:, :, :, 1].astype(np.uint8).ravel()
    test_cls = (test_preds[:, :, :, 1] >= threshold).astype(np.uint8).ravel()
    test_true = y_test[:, :, :, 1].astype(np.uint8).ravel()

    train_f1 = f1_score(train_true, train_cls, zero_division=0)
    test_f1 = f1_score(test_true, test_cls, zero_division=0)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    val_loss = history.history.get('val_loss', [])
    val_acc = history.history.get('val_accuracy', [])
    val_loss_trend = 'down' if len(val_loss) > 1 and val_loss[-1] <= val_loss[0] else 'up_or_flat'
    val_acc_trend = 'up' if len(val_acc) > 1 and val_acc[-1] >= val_acc[0] else 'down_or_flat'

    full_prob = predict_full_probability(
        model,
        source_images[0],
        args.tile,
        args.batch,
        stride=args.infer_stride
    )
    gt_full = (source_masks[0] == 255).astype(np.uint8).ravel()
    full_threshold = threshold
    full_macro_f1 = f1_score(gt_full, (full_prob >= full_threshold).astype(np.uint8).ravel(), average='macro', zero_division=0)
    if args.full_threshold_sweep:
        full_threshold, full_macro_f1 = pick_threshold_by_full_macro_f1(
            gt_full,
            full_prob,
            low=args.full_thr_low,
            high=args.full_thr_high,
            points=args.full_thr_points
        )

    full_prediction = (full_prob >= full_threshold).astype(np.uint8) * 255
    full_prediction = clean_binary_mask(full_prediction, min_size=args.clean_min_size)
    io.imsave(os.path.join(args.out, 'result.png'), full_prediction)
    save_patch_predictions(
        test_preds,
        args.out,
        threshold=threshold,
        clean_min_size=args.clean_min_size,
        limit=3
    )

    with open(os.path.join(args.out, 'metrics.txt'), 'w', encoding='utf-8') as metrics_file:
        metrics_file.write(f'train F1 {train_f1}\n')
        metrics_file.write(f'test F1 {test_f1}\n')
        metrics_file.write(f'test loss {test_loss}\n')
        metrics_file.write(f'test acc {test_acc}\n')
        metrics_file.write(f'threshold {threshold}\n')
        metrics_file.write(f'full threshold {full_threshold}\n')
        metrics_file.write(f'full macro F1 {full_macro_f1}\n')
        metrics_file.write(f'val best F1 {val_best_f1}\n')
        metrics_file.write(f'val_loss trend {val_loss_trend}\n')
        metrics_file.write(f'val_accuracy trend {val_acc_trend}\n')

    pred_full = (full_prediction > 0).astype(np.uint8).ravel()
    report_text = classification_report(
        gt_full,
        pred_full,
        target_names=['Background', 'Object'],
        zero_division=0
    )
    with open(os.path.join(args.out, 'report.txt'), 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

    print('train F1:', train_f1)
    print('test F1:', test_f1)
    print('test loss:', test_loss, 'test acc:', test_acc)
    print('val_loss trend:', val_loss_trend)
    print('val_accuracy trend:', val_acc_trend)
    print('results saved to:', args.out)
