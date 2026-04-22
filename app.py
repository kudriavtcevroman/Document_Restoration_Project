import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import zipfile
import io
import tempfile
import logging
from celery import Celery
import gradio as gr
from torch.amp import autocast
import shutil
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

celery_app = Celery('app',
                    broker='redis://localhost:6379/0',
                    backend='redis://localhost:6379/1',
                    broker_connection_retry_on_startup=True,
                    worker_pool='solo')

# ====================== РАБОЧЕЕ РАЗРЕШЕНИЕ ======================
class PaddedResize:
    def __init__(self, size=(1448, 1024), div_factor=8):
        self.size = size
        self.div_factor = div_factor
    def __call__(self, img):
        img = transforms.Resize(self.size, interpolation=Image.LANCZOS)(img)
        w, h = img.size
        pad_w = (self.div_factor - w % self.div_factor) % self.div_factor
        pad_h = (self.div_factor - h % self.div_factor) % self.div_factor
        if pad_w or pad_h:
            img = transforms.Pad((0, 0, pad_w, pad_h), fill=0)(img)
        return img

pred_transform = transforms.Compose([
    PaddedResize(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

restore_transform = transforms.ToTensor()

def pad_to_multiple(tensor, multiple=32):
    _, _, h, w = tensor.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    return F.pad(tensor, (0, pw, 0, ph), mode='reflect'), (ph, pw)

def unpad(tensor, ph, pw):
    return tensor[:, :, :tensor.shape[2] - ph, :tensor.shape[3] - pw]

# ====================== МОДЕЛИ ======================
pred_model = models.resnet50()
pred_model.fc = nn.Linear(pred_model.fc.in_features, 6)
state_dict = torch.load('prediction_model.pth', map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('fc.1.'):
        new_k = k.replace('fc.1.', 'fc.')
        new_state_dict[new_k] = v
    else:
        new_state_dict[k] = v
pred_model.load_state_dict(new_state_dict, strict=False)
pred_model = pred_model.to(device).eval()

# Restormer bad_print
sys.path.append('Restormer/basicsr/models/archs')
from restormer_arch import Restormer
restormer_badprint = Restormer(inp_channels=3, out_channels=3, dim=48,
                               num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                               heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False)
state = torch.load('finetuned_restormer_bad_print.pth', map_location=device)
if isinstance(state, dict) and 'model_state_dict' in state:
    restormer_badprint.load_state_dict(state['model_state_dict'])
else:
    restormer_badprint.load_state_dict(state)
restormer_badprint = restormer_badprint.to(device).eval()

# Restormer pixelation
restormer_pixelation = Restormer(inp_channels=3, out_channels=3, dim=48,
                                 num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                                 bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False)
state = torch.load('finetuned_restormer_pixelation.pth', map_location=device)
if isinstance(state, dict) and 'model_state_dict' in state:
    restormer_pixelation.load_state_dict(state['model_state_dict'])
else:
    restormer_pixelation.load_state_dict(state)
restormer_pixelation = restormer_pixelation.to(device).eval()

# EnlightenGAN
sys.path.append('EnlightenGAN')
from models.networks import define_G

class Opt:
    def __init__(self):
        self.self_attention = False
        self.use_norm = 1
        self.syn_norm = False
        self.use_avgpool = 0
        self.tanh = False
        self.times_residual = False
        self.linear_add = False
        self.latent_threshold = False
        self.latent_norm = False
        self.linear = False
        self.skip = 1.0

opt = Opt()
gpu_ids = [0] if device.type == 'cuda' else []
enlightengan = define_G(input_nc=3, output_nc=3, ngf=160, which_model_netG='sid_unet_resize',
                        norm='batch', skip=opt.skip, opt=opt, gpu_ids=gpu_ids)
enlightengan.load_state_dict(torch.load('finetuned_enlightengan.pth', map_location=device), strict=False)
enlightengan = enlightengan.to(device).eval()

# ====================== КЛАССЫ ======================
class_names = ['background', 'bad_print', 'brightness_contrast', 'clean', 'not_document', 'pixelation']

# ====================== CELERY TASK ======================
@celery_app.task
def process_single_image(image_bytes: bytes, filename: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_in:
            tmp_in.write(image_bytes)
            tmp_in_path = tmp_in.name

        img = Image.open(tmp_in_path).convert('RGB')

        pred_tensor = pred_transform(img).unsqueeze(0).to(device)
        restore_tensor = restore_transform(img).unsqueeze(0).to(device)

        restore_padded, (ph, pw) = pad_to_multiple(restore_tensor, multiple=32)

        with torch.no_grad():
            pred = pred_model(pred_tensor)
            dist_idx = torch.argmax(pred, dim=1).item()
            distortion = class_names[dist_idx]

        with torch.no_grad():
            if distortion == 'bad_print':
                restored_padded = restormer_badprint(restore_padded)
            elif distortion == 'brightness_contrast':
                gray_padded = restore_padded.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                restored_padded = enlightengan(restore_padded, gray_padded)[0]
            elif distortion == 'pixelation':
                restored_padded = restormer_pixelation(restore_padded)
            elif distortion == 'background':
                input_name = os.path.basename(tmp_in_path)
                base = os.path.splitext(input_name)[0]
                shutil.copy(tmp_in_path, os.path.join('DocScanner', "distorted", input_name))
                subprocess.run(["python", "inference.py"], cwd='DocScanner', check=True)
                restored_path = os.path.join('DocScanner', "rectified", f"{base}_rec.png")
                if not os.path.exists(restored_path):
                    restored_path = os.path.join('DocScanner', "rectified", input_name)
                restored_img = Image.open(restored_path).convert('RGB')

                # ИСПРАВЛЕНИЕ: сохраняем в temp-файл и возвращаем путь (Celery требует строку)
                out_path = tempfile.mktemp(suffix='.png')
                restored_img.save(out_path)

                os.remove(tmp_in_path)
                return filename, distortion, 'Восстановлено', out_path
            else:
                restored_padded = restore_padded

            restored = unpad(restored_padded, ph, pw)

        restored_img = transforms.ToPILImage()(restored.squeeze(0).clamp(0, 1).cpu())
        out_path = tempfile.mktemp(suffix='.png')
        restored_img.save(out_path)

        status = 'Восстановлено' if distortion not in ['clean', 'not_document'] else 'Без изменений'
        os.remove(tmp_in_path)
        return filename, distortion, status, out_path

    except Exception as e:
        logging.error(f"Ошибка обработки {filename}: {e}")
        raise

# ====================== GRADIO ======================
def process_input(file_obj, progress=gr.Progress()):
    if file_obj is None:
        return None, "Файл не загружен"

    results = []
    with open(file_obj.name, 'rb') as f:
        file_bytes = f.read()

    if zipfile.is_zipfile(io.BytesIO(file_bytes)):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zin:
            valid_files = [name for name in zin.namelist() if name.lower().endswith(('.jpg', '.png'))]
            if not valid_files:
                return None, "В архиве нет jpg/png файлов"

            out_zip_path = tempfile.mktemp(suffix='.zip')
            with zipfile.ZipFile(out_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
                tasks = []
                for name in valid_files:
                    img_data = zin.read(name)
                    task = process_single_image.delay(img_data, name)
                    tasks.append((name, task))

                progress(0, desc="Обработка ZIP...")
                for i, (name, task) in enumerate(tasks):
                    try:
                        fn, dist, stat, path = task.get(timeout=900)
                        results.append(f"{fn}: {dist} — {stat}")
                        with open(path, 'rb') as rf:
                            zout.writestr(fn, rf.read())
                        os.remove(path)
                    except Exception as e:
                        results.append(f"{name}: Ошибка — {str(e)}")
                    progress((i + 1) / len(tasks))

            return out_zip_path, "\n".join(results)

    else:
        filename = os.path.basename(file_obj.name)
        if not filename.lower().endswith(('.jpg', '.png')):
            return None, "Поддерживаются только JPG/PNG"
        try:
            fn, dist, stat, path = process_single_image.delay(file_bytes, filename).get(timeout=900)
            return path, f"{fn}: {dist} — {stat}"
        except Exception as e:
            return None, f"Ошибка: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Восстановление документов")
    input_file = gr.File(label="JPG/PNG или ZIP")
    output_file = gr.File(label="Результат")
    output_log = gr.Textbox(label="Лог", lines=10)
    gr.Button("Обработать").click(process_input, inputs=[input_file], outputs=[output_file, output_log])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inline=False)