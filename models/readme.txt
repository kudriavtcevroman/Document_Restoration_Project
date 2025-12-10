Эта директория предназначена для хранения весов обученных моделей. Она пуста по умолчанию — скачайте файлы по ссылкам ниже и разместите здесь.

## Назначение:
- Веса используются для инференса в предиктивной модели и моделях восстановления.
- Формат: .pth (PyTorch).

## Инструкции по скачиванию:
- prediction_model.pth: Веса предиктивной модели (ResNet50) для классификации искажений.
  Ссылка: https://drive.google.com/file/d/12aPG4dFQ_r64eg0e-yo1ak3yG4wYjU7H/view?usp=sharing
- finetuned_restormer.pth: Веса fine-tuned Restormer для "плохой печати".
  Ссылка: https://drive.google.com/file/d/1QUgPn7qs0kHj8M8lmmgtIy44qVxvMy6a/view?usp=sharing
- finetuned_real_esrgan.pth: Веса fine-tuned Real-ESRGAN для "пикселизации".
  Ссылка: https://drive.google.com/file/d/1lgTLQ3K-q52WRmXmdLYb60kWdaJJjHvv/view?usp=sharing
- finetuned_enlightengan.pth: Веса fine-tuned EnlightenGAN для "засветления/затемнения".
  Ссылка: https://drive.google.com/file/d/1kkTi8dWub0jL7zA10B_0VDtEzGX0GpME/view?usp=sharing

## Использование:
- Загружайте в модели: model.load_state_dict(torch.load('prediction_model.pth'))
- Убедитесь в наличии GPU.
- Если репозиторий клонирован локально, используйте gdown для автоматизации скачивания (пример в server_colab.ipynb).
