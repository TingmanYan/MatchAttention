import gradio as gr
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os
import time

from dataloader.stereo import transforms
from utils.utils import InputPadder, calc_noc_mask
from models.match_stereo import MatchStereo

torch.backends.cudnn.benchmark = True

class MatchStereoDemo:
    def __init__(self):
        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if self.has_cuda else 'cpu'
        self.model = None
        self.current_variant = None
        self.current_mode = None
        self.current_precision = None
        self.current_mat_impl = None
        
    def load_model(self, mode, variant, precision, mat_impl):
        """load model, skip if the model has been loaded"""
        if (self.model is not None and 
            self.current_variant == variant and 
            self.current_mode == mode and
            self.current_precision == precision and
            self.current_mat_impl == mat_impl):
            return "Model already loaded"
            
        # fixed checkpoint path
        checkpoint_base_path = "./checkpoints"
        if mode == 'stereo':
            checkpoint_name = f"match{mode}_{variant}_fsd.pth"
        elif mode == 'flow':
            checkpoint_name = f"match{mode}_{variant}_sintel.pth"
        else:
            raise NotImplementedError

        checkpoint_path = os.path.join(checkpoint_base_path, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            return f"Error: Checkpoint not found at {checkpoint_path}"
        
        args = argparse.Namespace()
        args.mode = mode
        args.variant = variant
        args.mat_impl = mat_impl
        
        if not self.has_cuda:
            precision = "fp32"
        dtypes = {'fp32': torch.float32, 'fp16': torch.float16}
        self.dtype = dtypes[precision]
        
        self.model = MatchStereo(args)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict=checkpoint['model'], strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.model = self.model.to(self.dtype)
            
            self._warmup_model()
            
            self.current_variant = variant
            self.current_mode = mode
            self.current_precision = precision
            self.current_mat_impl = mat_impl
            
            device_info = "GPU" if self.has_cuda else "CPU"
            return f"Successfully loaded {mode} {variant} model on {device_info} (precision: {precision}, mat_impl: {mat_impl})"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def _warmup_model(self):
        """warmup the model for accurate time measurement"""
        if self.model is None:
            return
            
        dummy_left = torch.randn(1, 3, 512, 512, device=self.device, dtype=self.dtype)
        dummy_right = torch.randn(1, 3, 512, 512, device=self.device, dtype=self.dtype)
        
        with torch.no_grad():
            _ = self.model(dummy_left, dummy_right, stereo=(self.current_mode == 'stereo'))
    
    def run_frame(self, left, right, stereo, low_res_init=False, factor=2.):
        """single frame inference"""
        if low_res_init:
            left_ds = F.interpolate(left, scale_factor=1/factor, mode='bilinear', align_corners=True)
            right_ds = F.interpolate(right, scale_factor=1/factor, mode='bilinear', align_corners=True)
            padder_ds = InputPadder(left_ds.shape, padding_factor=32)
            left_ds, right_ds = padder_ds.pad(left_ds, right_ds)

            field_up_ds = self.model(left_ds, right_ds, stereo=stereo)['field_up']
            field_up_ds = padder_ds.unpad(field_up_ds.permute(0, 3, 1, 2).contiguous()).contiguous()
            field_up_init = F.interpolate(field_up_ds, scale_factor=factor/32, mode='bilinear', align_corners=True)*(factor/32)
            field_up_init = field_up_init.permute(0, 2, 3, 1).contiguous()
            results_dict = self.model(left, right, stereo=stereo, init_flow=field_up_init)
        else:
            results_dict = self.model(left, right, stereo=stereo)

        return results_dict
    
    def get_inference_size(self, size_name):
        if size_name == "Original":
            return None
        
        def round_to_32(x):
            return (x + 16) // 32 * 32
        
        size_presets = {
            "720P": (round_to_32(1280), round_to_32(720)),
            "1080P": (round_to_32(1920), round_to_32(1080)),
            "2K": (round_to_32(2048), round_to_32(1080)),
            "4K UHD": (round_to_32(3840), round_to_32(2160))
        }
        
        return size_presets.get(size_name, None)
    
    def process_images(self, left_image, right_image, mode, variant, 
                      low_res_init=False, inference_size_name="Original", 
                      precision="fp32", mat_impl="pytorch"):
        if not self.has_cuda:
            precision = "fp32"
            mat_impl = "pytorch"
        
        load_result = self.load_model(mode, variant, precision, mat_impl)
        if load_result.startswith("Error"):
            return None, None, None, load_result
        
        try:
            left = np.array(left_image.convert('RGB')).astype(np.float32)
            right = np.array(right_image.convert('RGB')).astype(np.float32)
            
            original_size = left.shape[:2]  # (H, W)
            
            inference_size = self.get_inference_size(inference_size_name)
            
            val_transform_list = [transforms.ToTensor(no_normalize=True)]
            val_transform = transforms.Compose(val_transform_list)
            
            sample = {'left': left, 'right': right}
            sample = val_transform(sample)
            left_tensor = sample['left'].to(self.device, dtype=self.dtype).unsqueeze(0)
            right_tensor = sample['right'].to(self.device, dtype=self.dtype).unsqueeze(0)
            
            stereo = (mode == 'stereo')
            
            ori_size = left_tensor.shape[-2:]
            if inference_size is not None:
                left_tensor = F.interpolate(left_tensor, size=inference_size, mode='bilinear', align_corners=True)
                right_tensor = F.interpolate(right_tensor, size=inference_size, mode='bilinear', align_corners=True)
                padder = None
            else:
                padder = InputPadder(left_tensor.shape, padding_factor=32)
                left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
            
            device_type = "GPU" if self.has_cuda else "CPU"
            actual_size = inference_size if inference_size else ori_size
            status_info = f"Device: {device_type} | Resolution: {actual_size[1]}x{actual_size[0]} | Precision: {precision}"
            
            start_time = time.time()
            with torch.no_grad():
                results_dict = self.run_frame(left_tensor, right_tensor, stereo, low_res_init)
            inference_time = (time.time() - start_time) * 1000 # ms
            
            field_up = results_dict['field_up'].permute(0, 3, 1, 2).float().contiguous()
            
            if padder is not None:
                field_up = padder.unpad(field_up)
            elif inference_size is not None:
                field_up = F.interpolate(field_up, size=ori_size, mode='bilinear', align_corners=True)
                field_up[:, 0] = field_up[:, 0] * (ori_size[1] / float(inference_size[1]))
                field_up[:, 1] = field_up[:, 1] * (ori_size[0] / float(inference_size[0]))
            
            noc_mask = calc_noc_mask(field_up.permute(0, 2, 3, 1), A=8)
            noc_mask = noc_mask[0].detach().cpu().numpy()
            noc_mask = np.where(noc_mask, 255, 128).astype(np.uint8)
            
            field_up = torch.cat((field_up, torch.zeros_like(field_up[:, :1])), dim=1)
            field_up = field_up.permute(0, 2, 3, 1).contiguous()
            field, field_r = field_up.chunk(2, dim=0)
            
            if stereo:
                disparity = (-field[..., 0]).clamp(min=0)
                
                disparity_np = disparity[0].detach().cpu().numpy()
                min_val = disparity_np.min()
                max_val = disparity_np.max()
                if max_val - min_val > 1e-6:
                    disparity_norm = (disparity_np - min_val) / (max_val - min_val)
                else:
                    disparity_norm = np.zeros_like(disparity_np)
                disparity_img = (disparity_norm * 255).astype(np.uint8)
                
                return disparity_img, noc_mask, f"Inference time: {inference_time:.2f} ms. (Please re-run to get accurate time.)", status_info
            else:
                flow = field[0].detach().cpu().numpy()
                flow_rgb = self.flow_to_color(flow)
                return flow_rgb, noc_mask, f"Inference time: {inference_time:.2f} ms. (Please re-run to get accurate time.)", status_info
                
        except Exception as e:
            device_type = "GPU" if self.has_cuda else "CPU"
            return None, None, f"Error during inference: {str(e)}", f"Device: {device_type} | Error occurred"
    
    def flow_to_color(self, flow):
        """visualization of flow"""
        u = flow[..., 0]
        v = flow[..., 1]
        
        rad = np.sqrt(u**2 + v**2)
        rad_max = np.max(rad)
        epsilon = 1e-8
        
        if rad_max > epsilon:
            u = u / (rad_max + epsilon)
            v = v / (rad_max + epsilon)
        
        h, w = u.shape
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(u, v)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return flow_rgb

demo_model = MatchStereoDemo()

# example images
examples = [
    ["examples/booster_bathroom_left.png", "examples/booster_bathroom_right.png", "stereo", "tiny"],
    ["examples/staircase_q_left.png", "examples/staircase_q_right.png", "stereo", "tiny"],
    ["examples/frame_0031_clean.png", "examples/frame_0032_clean.png", "flow", "base"],
]

def process_inference(left_img, right_img, mode, variant, 
                     low_res_init, inference_size, precision, mat_impl):
    """Gradio function"""
    if left_img is None or right_img is None:
        return None, None, "Please upload both left and right images", "Waiting for input..."
    
    try:
        result = demo_model.process_images(
            left_img, right_img, mode, variant, 
            low_res_init, inference_size, precision, mat_impl
        )
        return result
    except Exception as e:
        return None, None, f"Error during inference: {str(e)}", f"Error: {str(e)}"

def update_variant_choices(mode):
    if mode == "flow":
        return gr.Radio(choices=["base"], value="base")
    else:
        return gr.Radio(choices=["tiny", "small", "base"], value="tiny")

# Gradio UI
with gr.Blocks(title="MatchStereo/MatchFlow Demo") as demo:
    gr.Markdown("# MatchStereo/MatchFlow Demo")
    gr.Markdown("Upload stereo images for disparity estimation or consecutive frames for optical flow estimation.")
    
    if not demo_model.has_cuda:
        gr.Markdown("> Note: Running on CPU. Some options (fp16, cuda) are disabled.")
    
    with gr.Row():
        with gr.Column():
            left_image = gr.Image(label="Left Image / Frame 1", type="pil")
            right_image = gr.Image(label="Right Image / Frame 2", type="pil")
            
            with gr.Row():
                mode = gr.Radio(
                    choices=["stereo", "flow"],
                    label="Mode",
                    value="stereo",
                    info="Select stereo for disparity estimation or flow for optical flow"
                )
                variant = gr.Radio(
                    choices=["tiny", "small", "base"],
                    label="Model Variant",
                    value="tiny",
                    info="Model size variant"
                )
            
            with gr.Row():
                low_res_init = gr.Checkbox(
                    label="Low Resolution Init",
                    value=False,
                    info="Use low-resolution initialization for high-res images (>=2K)"
                )
                inference_size = gr.Dropdown(
                    choices=["Original", "720P", "1080P", "2K", "4K UHD"],
                    label="Inference Size",
                    value="Original",
                    info="Rounded to multiples of 32"
                )
            
            with gr.Row():
                precision = gr.Radio(
                    choices=["fp32", "fp16"],
                    label="Precision",
                    value="fp32",
                    info="Model precision",
                    interactive=demo_model.has_cuda
                )
                mat_impl = gr.Radio(
                    choices=["cuda", "pytorch"],
                    label="MatchAttention Implementation",
                    value="cuda",
                    info="MatchAttention implementations",
                    interactive=demo_model.has_cuda
                )
            
            run_btn = gr.Button("Run Inference", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Output Result", interactive=False)
            noc_mask = gr.Image(label="NOC Mask", interactive=False)
            time_output = gr.Textbox(label="Inference Time", interactive=False)
            status = gr.Textbox(label="Status Info", interactive=False, lines=2)
    
    gr.Markdown("## Examples")
    gr.Examples(
        examples=examples,
        inputs=[left_image, right_image, mode, variant],
        outputs=[output_image, noc_mask, time_output, status],
        fn=process_inference,
        cache_examples=False,
        label="Click any example below to load it"
    )
    
    run_btn.click(
        fn=process_inference,
        inputs=[left_image, right_image, mode, variant, 
                low_res_init, inference_size, precision, mat_impl],
        outputs=[output_image, noc_mask, time_output, status]
    )
    
    mode.change(
        fn=update_variant_choices,
        inputs=[mode],
        outputs=[variant]
    )

if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        print("Please install OpenCV for optical flow visualization: pip install opencv-python")
    
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False
    )