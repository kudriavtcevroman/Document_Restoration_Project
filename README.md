# Document Restoration Project

Проект по восстановлению изображений документов с использованием DL-моделей. Включает предиктивную модель для классификации искажений и fine-tuned модели для исправления (Restormer, EnlightenGAN, Real-ESRGAN).

## Структура
- notebooks/: Ноутбуки для создания датасетов, обучения, тестирования.
- scripts/: Вспомогательные скрипты (эффекты искажений).
- models/: Место для весов моделей (скачайте по ссылкам ниже).
- data/: Место для датасетов (скачайте по ссылкам ниже).
- tests/: Для тестовых изображений.

## Ссылки на датасеты (Google Drive)
- https://drive.google.com/drive/folders/1pl5iSfgUZSFefeuP-HYj85ORMpPojSX5?usp=sharing

## Ссылки на веса моделей (Google Drive)
- prediction_model.pth: https://drive.google.com/file/d/12aPG4dFQ_r64eg0e-yo1ak3yG4wYjU7H/view?usp=sharing
- finetuned_restormer.pth: https://drive.google.com/file/d/1QUgPn7qs0kHj8M8lmmgtIy44qVxvMy6a/view?usp=sharing
- finetuned_real_esrgan.pth: https://drive.google.com/file/d/1lgTLQ3K-q52WRmXmdLYb60kWdaJJjHvv/view?usp=sharing
- finetuned_enlightengan.pth: https://drive.google.com/file/d/1kkTi8dWub0jL7zA10B_0VDtEzGX0GpME/view?usp=sharing

## Установка
1. Клонируйте репозиторий: `git clone https://github.com/yourusername/DocumentRestorationProject.git`
2. Установите зависимости: `pip install -r requirements.txt`
3. Скачайте веса и датасеты в соответствующие folders.

## Запуск сервера
Используйте notebooks/server_colab.ipynb в Google Colab для запуска сервера с Gradio и Celery.
- Откройте в Colab.
- Установите runtime с GPU.
- Запустите все cells.
- Gradio предоставит публичный URL для доступа.

## Описание сервера
- Загрузка: Изображение или ZIP-архив (jpg/png, лимит 50MB).
- Обработка: Классификация искажений, восстановление.
- Вывод: Таблица результатов, скачивание обработанного.

![Пример аугментации документа моделью Restormer](assets/Restormer_augmentation.jpg)
![Пример аугментации документа моделью EnlightenGAN](assets/Enlightengan_augmentation.jpg)
![Пример аугментации документа моделью Real-ESRGAN](assets/Real-esrgan_augmentation.jpg)
