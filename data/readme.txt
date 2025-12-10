Эта директория предназначена для хранения датасетов проекта. Она пуста по умолчанию — скачайте файлы по ссылке ниже и разархивируйте здесь.

## Назначение:
- Датасеты используются для обучения и тестирования моделей.
- Включают подмножества из DocBank с искажениями (для fine-tuning) и для классификации.

## Инструкции по скачиванию:
- Полный набор датасетов на Google Drive: https://drive.google.com/drive/folders/1pl5iSfgUZSFefeuP-HYj85ORMpPojSX5?usp=sharing
  - DocBank_Subset_6000: Датасет для fine-tuning моделей восстановления (6000 изображений с парами clean/distorted).
  - DocBank_Subset_Prediction_4000: Датасет для обучения предиктивной модели (4000 изображений с классами искажений).
  - DocBank_Subset_Prediction_7000: Модифицированный датасет с добавлением 'not_document' (7000 изображений).

## Использование:
- Разархивируйте в поддиректории (e.g., data/DocBank_Subset_6000/train/clean).
- Ноутбуки (e.g., prediction_model_train.ipynb) ожидают структуру train/val.
- Исходный DocBank: Скачайте с https://doc-analysis.github.io/docbank-page/index.html для расширения.
