Эта директория предназначена для хранения весов обученных моделей. Она пуста по умолчанию — скачайте файлы по ссылкам ниже и разместите здесь.

## Назначение:
- Веса используются для инференса в предиктивной модели и моделях восстановления.
- Формат: .pth (PyTorch).

## Инструкции по скачиванию:
- prediction_model.pth: Веса предиктивной модели (ResNet50) для классификации искажений.
  Ссылка: https://drive.google.com/file/d/1ucc7-Tqb317qb9-BnyxHNjBm6yeSJHBh/view?usp=sharing
- finetuned_restormer_bad_print.pth: Веса fine-tuned Restormer для "плохой печати".
  Ссылка: https://drive.google.com/file/d/1QUgPn7qs0kHj8M8lmmgtIy44qVxvMy6a/view?usp=sharing
- finetuned_restormer_pixelation.pth: Веса fine-tuned Real-ESRGAN для "пикселизации".
  https://drive.google.com/file/d/1cHBAqs1PjuxxzKO4TS2c-C5p-zDSb3ae/view?usp=sharing
- finetuned_enlightengan.pth: Веса fine-tuned Restormer для "засветления/затемнения".
  Ссылка: https://drive.google.com/file/d/1kkTi8dWub0jL7zA10B_0VDtEzGX0GpME/view?usp=sharing
- DocScanner-L.pth и seg.pth: Веса DocScanner для "документов с фоном".
  Ссылка: https://drive.google.com/file/d/1oEpjD1eSOAf_BPfRYqfZwfaR6ZtWOBU8/view?usp=sharing
  Ссылка: https://drive.google.com/file/d/1Ik59a5iQ0stXZAxv6rsdjj4TB0Mxi3-p/view?usp=sharing

## Использование:
- Загружайте модели: model.load_state_dict(torch.load('prediction_model.pth'))
- Убедитесь в наличии GPU.
- Если репозиторий клонирован локально, используйте gdown для автоматизации скачивания.
