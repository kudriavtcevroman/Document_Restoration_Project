"""
Эффекты для искажения изображений.

Этот модуль содержит функции для применения реалистичных искажений к изображениям документов:
- плохая печать (с полосами, пятнами и шумом),
- старение (пятна от времени),
- засветление/затемнение (области с изменением яркости/контраста),
- пикселизация (масштабирование с nearest neighbor интерполяцией).

Параметры функций масштабированы относительно размера изображения для сохранения эффекта
на высоком разрешении (например, 1024x1448). Используются OpenCV и NumPy.

Пример использования:
    import cv2
    from effects import apply_bad_print_effect

    img = cv2.imread('image.jpg')
    distorted = apply_bad_print_effect(img)
    cv2.imwrite('distorted.jpg', distorted)
"""

import cv2
import numpy as np
import random


def apply_bad_print_effect(image):
    """
    Применяет эффект плохой печати: случайные полосы (горизонтальные/вертикальные) с затемнением,
    чернильные пятна и шум. Параметры масштабированы относительно размера изображения для сохранения заметности
    на высоком разрешении (например, ширина полосы ~0.1-0.5% от ширины/высоты).

    Args:
        image (np.ndarray): Входное изображение в формате NumPy array (RGB).

    Returns:
        np.ndarray: Искаженное изображение (RGB).
    """
    h, w = image.shape[:2]
    bad_print = image.copy()

    # Количество полос: пропорционально размеру (больше для больших изображений)
    num_stripes = random.randint(int(h / 500), int(h / 200))

    for _ in range(num_stripes):
        if random.choice([True, False]):  # Вертикальная полоса
            x = random.randint(0, w)
            width = random.randint(max(1, int(w * 0.001)), max(5, int(w * 0.005)))
            stripe_length = random.randint(int(h * 0.2), h)
            y_start = random.randint(0, h - stripe_length)
            y_end = y_start + stripe_length
            bad_print[y_start:y_end, x:x + width] = bad_print[y_start:y_end, x:x + width] * random.uniform(0.3, 0.7)
        else:  # Горизонтальная полоса
            y = random.randint(0, h)
            height = random.randint(max(1, int(h * 0.001)), max(5, int(h * 0.005)))
            stripe_length = random.randint(int(w * 0.2), w)
            x_start = random.randint(0, w - stripe_length)
            x_end = x_start + stripe_length
            bad_print[y:y + height, x_start:x_end] = bad_print[y:y + height, x_start:x_end] * random.uniform(0.3, 0.7)

    # Количество чернильных пятен: пропорционально площади
    num_ink_spots = random.randint(int(h * w / 100000), int(h * w / 20000))
    for _ in range(num_ink_spots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        radius = random.randint(max(1, int(min(h, w) * 0.001)), max(5, int(min(h, w) * 0.005)))
        cv2.circle(bad_print, (x, y), radius, (0, 0, 0), -1)

    # Шум: стандартное отклонение пропорционально
    noise_std = random.uniform(5, 15) * (min(h, w) / 512)
    noise = np.random.normal(0, noise_std, bad_print.shape).astype(np.uint8)
    bad_print = cv2.add(bad_print, noise)

    return np.clip(bad_print, 0, 255).astype(np.uint8)


def apply_aging_effect(image):
    """
    Применяет эффект старения: случайные пятна затемнения (имитация пятен от времени).
    Параметры масштабированы: радиус пятен ~1-5% от минимального размера.

    Args:
        image (np.ndarray): Входное изображение в формате NumPy array (RGB).

    Returns:
        np.ndarray: Искаженное изображение (RGB).
    """
    h, w = image.shape[:2]
    aged = image.copy().astype(np.float32)  # Для точных расчётов

    # Количество пятен: пропорционально размеру
    num_spots = random.randint(int(min(h, w) / 200), int(min(h, w) / 100))

    for _ in range(num_spots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        radius = random.randint(max(5, int(min(h, w) * 0.01)), max(10, int(min(h, w) * 0.05)))

        # Маска пятна с градиентом
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        spot_mask = np.exp(-(xx ** 2 + yy ** 2) / (2 * (radius / 3) ** 2))
        spot_mask = np.clip(spot_mask, 0, 1)

        # Вставка маски с учётом границ
        mask = np.zeros((h, w), dtype=np.float32)
        y_start = max(0, y - radius)
        y_end = min(h, y + radius)
        x_start = max(0, x - radius)
        x_end = min(w, x + radius)
        mask[y_start:y_end, x_start:x_end] = spot_mask[:y_end - y_start, :x_end - x_start]

        spot_intensity = random.uniform(0.5, 0.8)
        aged = aged * (1 - mask[..., None]) + aged * spot_intensity * mask[..., None]

    return np.clip(aged, 0, 255).astype(np.uint8)


def apply_brightness_contrast(image, alpha_range=(0.8, 1.2), beta_range=(-30, 30), min_area_ratio=1/8, blur_kernel=21):
    """
    Применяет эффект засветления/затемнения: случайные области с изменением яркости/контраста.
    Параметры областей и ядра размытия масштабированы относительно размера (blur_kernel ~3-5% от min размера).

    Args:
        image (np.ndarray): Входное изображение в формате NumPy array (RGB).
        alpha_range (tuple): Диапазон для коэффициента контраста (по умолчанию (0.8, 1.2)).
        beta_range (tuple): Диапазон для смещения яркости (по умолчанию (-30, 30)).
        min_area_ratio (float): Минимальная доля площади области (по умолчанию 1/8).
        blur_kernel (int): Начальный размер ядра размытия (масштабируется).

    Returns:
        np.ndarray: Искаженное изображение (RGB).
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    # Количество областей: фиксировано, но формы случайные
    num_areas = random.randint(1, 4)
    for _ in range(num_areas):
        area_ratio = random.uniform(min_area_ratio, 1.0)
        area_w = int(w * area_ratio)
        area_h = int(h * area_ratio)
        x = random.randint(0, w - area_w)
        y = random.randint(0, h - area_h)

        shape = random.choice(["rect", "ellipse", "polygon"])
        if shape == "rect":
            mask[y:y + area_h, x:x + area_w] += 1.0
        elif shape == "ellipse":
            cv2.ellipse(mask, (x + area_w // 2, y + area_h // 2), (area_w // 2, area_h // 2), 0, 0, 360, 1.0, -1)
        else:  # polygon
            points = np.array([[random.randint(x, x + area_w), random.randint(y, y + area_h)] for _ in range(5)], np.int32)
            cv2.fillPoly(mask, [points], 1.0)

    # Размытие маски: ядро пропорционально
    blur_kernel = max(blur_kernel, int(min(h, w) * 0.03)) | 1  # Нечётное, ~3% min размера
    mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    mask = np.clip(mask, 0, 1)

    # Изменение яркости/контраста
    brightened = cv2.convertScaleAbs(image, alpha=random.uniform(1.1, 1.3), beta=random.randint(0, 20))
    darkened = cv2.convertScaleAbs(image, alpha=random.uniform(0.5, 0.9), beta=random.randint(-50, -10))

    adjusted = (image * (1 - mask[..., None]) + brightened * mask[..., None]).astype(np.uint8)
    adjusted = np.where(mask[..., None] < 0.5, darkened, adjusted)

    return adjusted


def apply_pixelation(image, min_scale=1.1, max_scale=1.4, step=0.05):
    """
    Применяет эффект пикселизации: уменьшение, затем увеличение с nearest neighbor.
    Масштаб оставлен, но для высокого разрешения эффект будет заметен, если scale >1.
    Для stronger эффекта на больших изображениях можно увеличить max_scale.

    Args:
        image (np.ndarray): Входное изображение в формате NumPy array (RGB).
        min_scale (float): Минимальный коэффициент масштабирования (по умолчанию 1.1).
        max_scale (float): Максимальный коэффициент масштабирования (по умолчанию 1.4).
        step (float): Шаг для округления масштаба (по умолчанию 0.05).

    Returns:
        np.ndarray: Искаженное изображение (RGB).
    """
    h, w = image.shape[:2]
    scale = round(random.uniform(min_scale, max_scale) / step) * step
    new_w = int(w / scale)
    new_h = int(h / scale)
    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated