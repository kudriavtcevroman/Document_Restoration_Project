import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr
from celery import Celery
import shutil
import subprocess
import gc
from torch.amp import autocast
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

restoration_transform = transforms.Compose([
    PaddedResize(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

pred_model = models.resnet50()
num_ftrs = pred_model.fc.in_features
pred_model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 6))
state = torch.load('prediction_model.pth', map_location=device)
pred_model.load_state_dict(state if not isinstance(state, dict) else state.get('model_state_dict', state))
pred_model = pred_model.to(device).eval()

sys.path.append('Restormer/basicsr/models/archs')
from restormer_arch import Restormer
restormer_bad = Restormer(inp_channels=3, out_channels=3, dim=48,
                          num_blocks=[4,6,6,8], num_refinement_blocks=4,
                          heads=[1,2,4,8], ffn_expansion_factor=2.66,
                          bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False,
                          use_checkpoint=True).to(device).half()
state = torch.load('finetuned_restormer_bad_print.pth', map_location=device)
restormer_bad.load_state_dict(state if not isinstance(state, dict) else state.get('model_state_dict', state))
restormer_bad.eval()

restormer_pix = Restormer(inp_channels=3, out_channels=3, dim=48,
                          num_blocks=[4,6,6,8], num_refinement_blocks=4,
                          heads=[1,2,4,8], ffn_expansion_factor=2.66,
                          bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False,
                          use_checkpoint=True).to(device).half()
state = torch.load('finetuned_restormer_pixelation.pth', map_location=device)
restormer_pix.load_state_dict(state if not isinstance(state, dict) else state.get('model_state_dict', state))
restormer_pix.eval()

sys.path.append('EnlightenGAN')
from models.networks import define_G
class Opt: pass
opt = Opt()
opt.skip = 1.0
enlightengan = define_G(input_nc=4, output_nc=3, ngf=160,
                        which_model_netG='sid_unet_resize', norm='batch',
                        use_dropout=False, gpu_ids=[0], skip=opt.skip, opt=opt).to(device).half()
state = torch.load('finetuned_enlightengan.pth', map_location=device)
enlightengan.load_state_dict(state if not isinstance(state, dict) else state.get('model_state_dict', state), strict=False)
enlightengan.eval()

app = Celery('app', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')
app.conf.task_routes = {'app.process_single_image': {'queue': 'doc_restoration'}}

@app.task
def process_single_image(image_path):
    img_pil = Image.open(image_path).convert('RGB')
    pred_tensor = pred_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = pred_model(pred_tensor)
        dist_idx = torch.argmax(logits, dim=1).item()
        distortion = ['background', 'bad_print', 'brightness_contrast', 'clean', 'not_document', 'pixelation'][dist_idx]

    if distortion in ['clean', 'not_document']:
        processed_img = img_pil
    elif distortion == 'background':
        input_name = os.path.basename(image_path)
        base = os.path.splitext(input_name)[0]
        shutil.copy(image_path, os.path.join('DocScanner', "distorted", input_name))
        subprocess.run(["python", "inference.py"], cwd='DocScanner')
        restored_path = os.path.join('DocScanner', "rectified", f"{base}_rec.png")
        if not os.path.exists(restored_path):
            restored_path = os.path.join('DocScanner', "rectified", input_name)
        processed_img = Image.open(restored_path).convert('RGB')
    else:
        if distortion == 'bad_print':
            model = restormer_bad
            needs_gray = False
        elif distortion == 'pixelation':
            model = restormer_pix
            needs_gray = False
        else:
            model = enlightengan
            needs_gray = True
        processed_img = process_image(model, image_path, needs_gray)

    return {"processed_img": processed_img}

def process_image(model, img_path, needs_gray=False):
    img = Image.open(img_path).convert('RGB')
    input_tensor = restoration_transform(img).unsqueeze(0).to(device)
    if needs_gray:
        gray_tensor = transforms.Grayscale(num_output_channels=1)(input_tensor)
        with torch.inference_mode(), autocast(device_type='cuda', dtype=torch.float16):
            output = model(input_tensor, gray_tensor)[0]
    else:
        with torch.inference_mode(), autocast(device_type='cuda', dtype=torch.float16):
            output = model(input_tensor)
    output_img = transforms.ToPILImage()(output.squeeze(0).clamp(0, 1).cpu())
    return output_img

def restore_document(image):
    temp_path = "temp_input.jpg"
    image.save(temp_path)
    task = process_single_image.delay(temp_path)
    result = task.get()
    return result["processed_img"]

with gr.Blocks(title="Document Restoration") as demo:
    gr.Markdown("# Document Restoration Project")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Загрузите изображение документа")
            btn = gr.Button("Восстановить документ")
        with gr.Column():
            output_img = gr.Image(type="pil", label="Результат восстановления")
    btn.click(restore_document, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
