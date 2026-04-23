# Document Restoration Project

Проект по восстановлению изображений документов с использованием DL-моделей.

Классифицирует искажения (ResNet50) и применяет модели:
- Restormer — эффекты "плохая печать" и "пикселизация"
- EnlightenGAN — эффект "плохой контраст"
- DocScanner — эффект "посторонний фон"

Поддержка JPG/PNG, лог искажений, сервер с Gradio + Celery (многопоточность через Redis).

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.0-green)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Структура репозитория
- `notebooks/` — Ноутбуки (датасеты, fine-tune, тесты).
- `scripts/` — py-файлы.
- `models/` — Веса моделей (скачайте по ссылкам).
- `data/` — Датасеты (скачайте по ссылкам).
- `metrics/` — Метрики обученных моделей.
- `assets/` — Тестовые изображения.
- `app.py` - py-файл для запуска приложения.
- `requirements.txt` - файл с зависимостями, необходимыми для работы приложения.
 
## Датасеты и веса (Google Drive)
- Датасеты: [папка](https://drive.google.com/drive/folders/1pl5iSfgUZSFefeuP-HYj85ORMpPojSX5?usp=sharing)
- Веса:
  - prediction_model.pth: [ссылка](https://drive.google.com/file/d/17NFLGU17VIebNaf0vCJsW238JXkm9k8o/view?usp=sharing)
  - finetuned_restormer_bad_print.pth: [ссылка](https://drive.google.com/file/d/1QUgPn7qs0kHj8M8lmmgtIy44qVxvMy6a/view?usp=sharing)
  - finetuned_restormer_pixelation.pth: [ссылка](https://drive.google.com/file/d/1cHBAqs1PjuxxzKO4TS2c-C5p-zDSb3ae/view?usp=sharing)
  - finetuned_enlightengan.pth: [ссылка](https://drive.google.com/file/d/1kkTi8dWub0jL7zA10B_0VDtEzGX0GpME/view?usp=sharing)
  - DocScanner-L.pth: [ссылка](https://drive.google.com/file/d/1oEpjD1eSOAf_BPfRYqfZwfaR6ZtWOBU8/view?usp=sharing)
  - seg.pth (для DocScanner): [ссылка](https://drive.google.com/file/d/1Ik59a5iQ0stXZAxv6rsdjj4TB0Mxi3-p/view?usp=sharing)

## Установка и запуск приложения (на Windows).
1. [Скачайте](https://github.com/redis/redis/releases) и установите Redis.

2. Запустите терминал и выполните:

```bash
git clone https://github.com/kudriavtcevroman/Document_Restoration_Project.git
cd Document_Restoration_Project

conda create -n doc_rest_app python=3.10 -y
conda activate doc_rest_app

git clone https://github.com/swz30/Restormer.git
git clone https://github.com/VITA-Group/EnlightenGAN.git
git clone https://github.com/fh2019ustc/DocScanner.git DocScanner

pip install -r requirements.txt
pip install "git+https://github.com/xinntao/BasicSR.git@8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a"

gdown https://drive.google.com/file/d/17NFLGU17VIebNaf0vCJsW238JXkm9k8o -O prediction_model.pth
gdown https://drive.google.com/uc?id=1QUgPn7qs0kHj8M8lmmgtIy44qVxvMy6a -O finetuned_restormer_bad_print.pth
gdown https://drive.google.com/file/d/1cHBAqs1PjuxxzKO4TS2c-C5p-zDSb3ae -O finetuned_restormer_pixelation.pth
gdown https://drive.google.com/uc?id=1kkTi8dWub0jL7zA10B_0VDtEzGX0GpME -O finetuned_enlightengan.pth
gdown https://drive.google.com/file/d/1oEpjD1eSOAf_BPfRYqfZwfaR6ZtWOBU8 -O DocScanner-L.pth
gdown https://drive.google.com/file/d/1Ik59a5iQ0stXZAxv6rsdjj4TB0Mxi3-p -O seg.pth

mkdir DocScanner\model_pretrained
move DocScanner-L.pth DocScanner\model_pretrained\DocScanner-L.pth
move seg.pth DocScanner\model_pretrained\seg.pth
```

3. Выполните проверку:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Если выдаст "False", то выполните:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

4. Запустите новый терминал и выполните:

```bash
cd C:\Program Files\Redis # Укажите путь к директории, куда установлен Redis

redis-server.exe
```

5. Запустите новый терминал и выполните:

```bash
cd Document_Restoration_Project

conda activate doc_rest_app

celery -A app.celery_app worker --loglevel=info --pool=solo # для многопоточного режима (celery -A app worker --loglevel=info --concurrency=2 --pool=threads)
```

6. Запустите новый терминал и выполните:

```bash
cd Document_Restoration_Project

conda activate doc_rest_app

python app.py
```

7. Откройте браузер и перейдите по адресу: http://127.0.0.1:7860



## Workflow обработки изображений в Document Restoration App

```mermaid
graph TD;
    A[Upload Image] --> B[Prediction Model];
    B --> C[Distorted Image];
    B --> D[Clean Image];
    B --> E[Not Document];
    C -->|bad print| F[Restormer];
    C -->|brightness contrast| G[EnlightenGAN];
    C -->|pixelation| H[Restormer];
    C -->|background| K[DocScaner];
    F --> I[Restored Image];
    G --> I[Restored Image];
    H --> I[Restored Image];
    K --> I[Restored Image];
    D --> J[Download Image];
    E --> J[Download Image];
    I --> J[Download Image];

    style A fill:#f9f,stroke:#333,color:#000,stroke-width:2px
    style B fill:#bbf,stroke:#333,color:#000,stroke-width:2px
    style C fill:#ff9,stroke:#333,color:#000,stroke-width:2px
    style D fill:#9f9,stroke:#333,color:#000,stroke-width:2px
    style E fill:#f99,stroke:#333,color:#000,stroke-width:2px
    style F fill:#9ff,stroke:#333,color:#000,stroke-width:2px
    style G fill:#9ff,stroke:#333,color:#000,stroke-width:2px
    style H fill:#9ff,stroke:#333,color:#000,stroke-width:2px
    style I fill:#9f9,stroke:#333,color:#000,stroke-width:2px
    style J fill:#f9f,stroke:#333,color:#000,stroke-width:2px
    style K fill:#9ff,stroke:#333,color:#000,stroke-width:2px
```

## Пример аугментации документа с эффектом плохой печати моделью Restormer_bad_print
![Пример аугментации документа моделью Restormer_bad_print](assets/Restormer_augmentation.jpg)

## Пример аугментации документа с плохим контрастом моделью EnlightenGAN
![Пример аугментации документа моделью EnlightenGAN](assets/Enlightengan_augmentation.jpg)

## Пример аугментации документа с плохим качеством моделью Restormer_pixelation
![Пример аугментации документа моделью Restormer_pixelation](assets/Real-esrgan_augmentation.jpg)

## Пример аугментации документа с посторонним фоном моделью DocScaner
![Пример аугментации документа моделью DocScaner](assets/Real-esrgan_augmentation.jpg)
