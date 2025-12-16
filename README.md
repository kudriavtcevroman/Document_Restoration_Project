Document Restoration Project

Проект по восстановлению изображений документов с использованием DL-моделей.

Классифицирует искажения (ResNet50) и применяет fine-tuned модели:
- Restormer — эффект "плохая печать"
- EnlightenGAN — эффект "плохой контраст"
- Real-ESRGAN — эффект "пикселизация"

Поддержка JPG/PNG, лог искажений, сервер с Gradio + Celery (многопоточность через Redis).

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.0-green)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Структура репозитория
- `notebooks/` — Ноутбуки (датасеты, fine-tune, тесты).
- `server/` — Сервер (app.py, requirements_server.txt).
- `scripts/` — effects.py (искажения).
- `models/` — Веса моделей (скачайте по ссылкам).
- `data/` — Датасеты (скачайте по ссылкам).
- `metrics/` — Метрики обученных моделей.
- `assets/` — Тестовые изображения.

## Датасеты и веса (Google Drive)
- Датасеты: [папка](https://drive.google.com/drive/folders/1pl5iSfgUZSFefeuP-HYj85ORMpPojSX5?usp=sharing)
- Веса:
  - prediction_model.pth: [ссылка](https://drive.google.com/file/d/12aPG4dFQ_r64eg0e-yo1ak3yG4wYjU7H/view?usp=sharing)
  - finetuned_restormer.pth: [ссылка](https://drive.google.com/file/d/1QUgPn7qs0kHj8M8lmmgtIy44qVxvMy6a/view?usp=sharing)
  - finetuned_real_esrgan.pth: [ссылка](https://drive.google.com/file/d/1lgTLQ3K-q52WRmXmdLYb60kWdaJJjHvv/view?usp=sharing)
  - finetuned_enlightengan.pth: [ссылка](https://drive.google.com/file/d/1kkTi8dWub0jL7zA10B_0VDtEzGX0GpME/view?usp=sharing)

Скачайте в корень или `models/`.

## Установка
```bash
git clone https://github.com/kudriavtcevroman/Document_Restoration_Project.git
cd Document_Restoration_Project

conda create -n doc_rest python=3.10 -y
conda activate doc_rest

git clone https://github.com/swz30/Restormer.git /content/Restormer
git clone https://github.com/VITA-Group/EnlightenGAN.git /content/EnlightenGAN
git clone https://github.com/xinntao/Real-ESRGAN.git /content/Real-ESRGAN

pip install -r requirements.txt
pip install "git+https://github.com/xinntao/BasicSR.git@8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a"
pip install -r server/requirements_server.txt  # Для сервера (Celery, Gradio)
```

```mermaid
graph TD;
    A[Upload Image] --> B[Prediction Model];
    B --> C[Distorted Image];
    B --> D[Clean Image];
    B --> E[Not Document];
    C -->|bad print| F[Restormer];
    C -->|brightness contrast| G[EnlightenGAN];
    C -->|pixelation| H[Real-ESRGAN];
    F --> I[Restored Image];
    G --> I[Restored Image];
    H --> I[Restored Image];
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
```
```mermaid
graph TD;
    A[Distorted Image] -->|Upload| B[Prediction Model];
    B -->|bad print| C[Restormer];
    B -->|brightness contrast| D[EnlightenGAN];
    B -->|pixelation| E[Real-ESRGAN];
    B -->|clean or not document| F[No restoration];

    style A fill:#f9f,stroke:#333,color:#000,stroke-width:2px
    style B fill:#bbf,stroke:#333,color:#000,stroke-width:2px
    style C fill:#ff9,stroke:#333,color:#000,stroke-width:2px
    style D fill:#9f9,stroke:#333,color:#000,stroke-width:2px
    style E fill:#f99,stroke:#333,color:#000,stroke-width:2px
    style F fill:#9ff,stroke:#333,color:#000,stroke-width:2px
```

## Пример аугментации документа с эффектом плохой печати моделью Restormer
![Пример аугментации документа моделью Restormer](assets/Restormer_augmentation.jpg)

## Пример аугментации документа с плохим контрастом моделью EnlightenGAN
![Пример аугментации документа моделью EnlightenGAN](assets/Enlightengan_augmentation.jpg)

## Пример аугментации документа с плохим качеством моделью Real-ESRGAN
![Пример аугментации документа моделью Real-ESRGAN](assets/Real-esrgan_augmentation.jpg)
