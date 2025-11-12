import cv2
import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from models.match_stereo import MatchStereo
from utils.utils import InputPadder
import time

import pyzed.sl as sl

# Global variables for storing disparity map information
current_disparity_map = None
current_disparity_colored = None
mouse_pressed = False
mouse_position = (0, 0)
left_rectified_mouse_pressed = False
left_rectified_mouse_position = (0, 0)
f_pixel = None
baseline_mm = None
last_inference_time = 0
fps = 0
display_width = 640
display_height = 360

# MatchStereo model for computing disparity
class MatchStereoDisparity:
    def __init__(self, args):
        self.device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() and args.device_id >=0 else 'cpu'
        dtypes = {'fp32': torch.float, 'fp16': torch.half, 'bf16': torch.bfloat16}
        self.dtype = dtypes[args.precision]
        
        self.model = MatchStereo(args)
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict=checkpoint['model'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.model = self.model.to(self.dtype)
        
        if torch.cuda.is_available() and not args.no_compile and args.device_id >=0:
            print('Compiling the model, this may take several minutes')
            torch.backends.cuda.matmul.allow_tf32 = True
            self.model = torch.compile(self.model, dynamic=False)
        
        self.stereo = (args.mode == 'stereo')
        self.low_res_init = args.low_res_init
        self.inference_size = args.inference_size
        
    def run_frame(self, left, right):
        if self.low_res_init: # downsample to 1/2, can also be 1/4
            factor = 2.
            left_ds = F.interpolate(left, scale_factor=1/factor, mode='bilinear', align_corners=True)
            right_ds = F.interpolate(right, scale_factor=1/factor, mode='bilinear', align_corners=True)
            padder_ds = InputPadder(left_ds.shape, padding_factor=32)
            left_ds, right_ds = padder_ds.pad(left_ds, right_ds)

            field_up_ds = self.model(left_ds, right_ds, stereo=self.stereo)['field_up']
            field_up_ds = padder_ds.unpad(field_up_ds.permute(0, 3, 1, 2).contiguous()).contiguous()
            field_up_init = F.interpolate(field_up_ds, scale_factor=factor/32, mode='bilinear', align_corners=True)*(factor/32) # init resolution 1/32
            field_up_init = field_up_init.permute(0, 2, 3, 1).contiguous()
            results_dict = self.model(left, right, stereo=self.stereo, init_flow=field_up_init)
        else:
            results_dict = self.model(left, right, stereo=self.stereo)

        return results_dict
    
    def compute_disparity(self, left_rectified, right_rectified):
        left_rectified = torch.from_numpy(left_rectified).permute(2, 0, 1).float()
        right_rectified = torch.from_numpy(right_rectified).permute(2, 0, 1).float()
        left_tensor = left_rectified.to(self.device, dtype=self.dtype).unsqueeze(0) # [1, 3, H, W]
        right_tensor = right_rectified.to(self.device, dtype=self.dtype).unsqueeze(0) # [1, 3, H, W]
        
        # Handle input size
        if self.inference_size is None:
            padder = InputPadder(left_tensor.shape, padding_factor=32)
            left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
            ori_size = None
        else:
            ori_size = left_tensor.shape[-2:]
            left_tensor = F.interpolate(left_tensor, size=self.inference_size, mode='bilinear', align_corners=True)
            right_tensor = F.interpolate(right_tensor, size=self.inference_size, mode='bilinear', align_corners=True)
        
        # Inference
        with torch.inference_mode():
            results_dict = self.run_frame(left_tensor, right_tensor)
            
            field_up = results_dict['field_up'].permute(0, 3, 1, 2).float().contiguous()
            
            if self.inference_size is None:
                field_up = padder.unpad(field_up)
            else:
                field_up = F.interpolate(field_up, size=ori_size, mode='bilinear', align_corners=True)
                field_up = field_up * ori_size[-1] / float(self.inference_size[-1])
            
            # Extract disparity map
            field_up = torch.cat((field_up, torch.zeros_like(field_up[:, :1])), dim=1)
            field_up = field_up.permute(0, 2, 3, 1).contiguous()
            field, _ = field_up.chunk(2, dim=0)
            field = field[0] # [H, W, 3]
            
            if self.stereo:
                disparity = (-field[..., 0]).clamp(min=0)
            
            disparity = disparity.detach().cpu().numpy()
        
        return disparity

class DisparityVisualizer:
    def __init__(self, min_percentile=2, max_percentile=98, adaptation_rate=0.1):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.adaptation_rate = adaptation_rate
        self.current_min = 0
        self.current_max = 500
        self.initialized = False
    
    def visualize_disparity_map(self, disparity_map):
        disparity_map_filtered = np.copy(disparity_map)
        disparity_map_filtered[disparity_map_filtered <= 0] = np.nan
        disparity_map_filtered[disparity_map_filtered > 800] = np.nan  # Filter out large disparity values
        
        valid_disparities = disparity_map_filtered[~np.isnan(disparity_map_filtered)]
        
        if len(valid_disparities) > 100:  # Ensure enough valid points
            if not self.initialized:
                # Initial setup
                self.current_min = np.percentile(valid_disparities, self.min_percentile)
                self.current_max = np.percentile(valid_disparities, self.max_percentile)
                self.initialized = True
            else:
                # Adaptive update of range
                frame_min = np.percentile(valid_disparities, self.min_percentile)
                frame_max = np.percentile(valid_disparities, self.max_percentile)
                
                self.current_min = (1 - self.adaptation_rate) * self.current_min + self.adaptation_rate * frame_min
                self.current_max = (1 - self.adaptation_rate) * self.current_max + self.adaptation_rate * frame_max
            
            # Ensure minimum range
            min_range = 50  # Minimum disparity range
            if self.current_max - self.current_min < min_range:
                center = (self.current_min + self.current_max) / 2
                self.current_min = center - min_range / 2
                self.current_max = center + min_range / 2
        
        disparity_vis = np.copy(disparity_map_filtered)
        disparity_vis[np.isnan(disparity_vis)] = 0
        
        # Normalize using current statistical range
        disparity_vis = np.clip(disparity_vis, self.current_min, self.current_max)
        disparity_vis = (disparity_vis - self.current_min) / (self.current_max - self.current_min) * 255
        
        disparity_vis = disparity_vis.astype(np.uint8)
        disparity_colored = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
        
        return disparity_colored, disparity_map_filtered

# Mouse callback function for displaying depth values in disparity map
def mouse_callback(event, x, y, flags, param):
    global mouse_pressed, mouse_position
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        mouse_position = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            mouse_position = (x, y)

# Mouse callback function for Left Rectified window
def left_rectified_mouse_callback(event, x, y, flags, param):
    global left_rectified_mouse_pressed, left_rectified_mouse_position
    
    if event == cv2.EVENT_LBUTTONDOWN:
        left_rectified_mouse_pressed = True
        left_rectified_mouse_position = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        left_rectified_mouse_pressed = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if left_rectified_mouse_pressed:
            left_rectified_mouse_position = (x, y)

# Calculate depth from disparity
def disparity_to_depth(disparity, f_pixel, baseline_mm):
    """
    Calculate depth from disparity
    Formula: depth = f * baseline / disparity
    where:
      - depth: depth value (mm)
      - f_pixel: focal length (pixels)
      - baseline_mm: baseline length (mm)
      - disparity: disparity value (pixels)
    """
    if disparity <= 0:
        return float('nan')
    
    # Calculate depth (mm)
    depth_mm = (f_pixel * baseline_mm) / disparity
    
    return depth_mm

# Convert display coordinates to original image coordinates
def display_to_original_coords(display_x, display_y, display_width, display_height, original_width, original_height):
    """
    Convert coordinates from display window to original image coordinates
    """
    scale_x = original_width / display_width
    scale_y = original_height / display_height
    
    original_x = int(display_x * scale_x)
    original_y = int(display_y * scale_y)
    
    # Ensure coordinates are within bounds
    original_x = max(0, min(original_x, original_width - 1))
    original_y = max(0, min(original_y, original_height - 1))
    
    return original_x, original_y

# Update disparity display including depth information at mouse position
def update_disparity_display(original_width, original_height):
    global current_disparity_map, current_disparity_colored, mouse_pressed, mouse_position, \
           left_rectified_mouse_pressed, left_rectified_mouse_position, f_pixel, baseline_mm, \
           last_inference_time, fps, display_width, display_height
    
    if current_disparity_map is not None and current_disparity_colored is not None:
        # Create a copy of the disparity map for display
        disparity_display = current_disparity_colored.copy()
        
        # Display resolution and performance info
        h, w = current_disparity_map.shape[:2]
        resolution_text = f"Resolution: {w}x{h}"
        inference_time_text = f"Inference: {last_inference_time:.3f}s"
        fps_text = f"FPS: {fps:.1f}"
        
        cv2.putText(disparity_display, resolution_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(disparity_display, inference_time_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(disparity_display, fps_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Check if we have a mouse event from Left Rectified window
        if left_rectified_mouse_pressed:
            # Convert display coordinates to original image coordinates
            orig_x, orig_y = display_to_original_coords(
                left_rectified_mouse_position[0], 
                left_rectified_mouse_position[1],
                display_width, display_height,
                original_width, original_height
            )
            
            # Check if coordinates are within valid range
            if 0 <= orig_y < current_disparity_map.shape[0] and 0 <= orig_x < current_disparity_map.shape[1]:
                disparity_value = current_disparity_map[orig_y, orig_x]
                
                # If disparity value is valid, calculate and display depth information
                if not np.isnan(disparity_value) and disparity_value > 0:
                    # Calculate depth
                    depth_value = disparity_to_depth(disparity_value, f_pixel, baseline_mm)
                    
                    # Convert back to display coordinates for disparity map
                    disp_x = int(orig_x * display_width / original_width)
                    disp_y = int(orig_y * display_height / original_height)
                    
                    # Draw crosshair and depth information on disparity map
                    cv2.line(disparity_display, (disp_x-10, disp_y), (disp_x+10, disp_y), (255, 255, 255), 2)
                    cv2.line(disparity_display, (disp_x, disp_y-10), (disp_x, disp_y+10), (255, 255, 255), 2)
                    
                    # Display disparity and depth value (mm only)
                    disparity_text = f"Disp: {disparity_value:.2f}px"
                    depth_text = f"Depth: {depth_value:.1f}mm"
                    
                    cv2.putText(disparity_display, disparity_text, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(disparity_display, depth_text, (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # If mouse is pressed in disparity map window, display depth information
        elif mouse_pressed:
            x, y = mouse_position
            
            # Convert display coordinates to original image coordinates
            orig_x, orig_y = display_to_original_coords(
                x, y,
                display_width, display_height,
                original_width, original_height
            )
            
            # Check if coordinates are within valid range
            if 0 <= orig_y < current_disparity_map.shape[0] and 0 <= orig_x < current_disparity_map.shape[1]:
                disparity_value = current_disparity_map[orig_y, orig_x]
                
                # If disparity value is valid, calculate and display depth information
                if not np.isnan(disparity_value) and disparity_value > 0:
                    # Calculate depth
                    depth_value = disparity_to_depth(disparity_value, f_pixel, baseline_mm)
                    
                    # Draw crosshair and depth information on disparity map
                    cv2.line(disparity_display, (x-10, y), (x+10, y), (255, 255, 255), 2)
                    cv2.line(disparity_display, (x, y-10), (x, y+10), (255, 255, 255), 2)
                    
                    # Display disparity and depth value (mm only)
                    disparity_text = f"Disp: {disparity_value:.2f}px"
                    depth_text = f"Depth: {depth_value:.1f}mm"
                    
                    cv2.putText(disparity_display, disparity_text, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(disparity_display, depth_text, (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Update displayed disparity map
        cv2.imshow("Disparity Map", disparity_display)

# Main program
def main():
    global current_disparity_map, current_disparity_colored, f_pixel, baseline_mm, last_inference_time, fps, \
           left_rectified_mouse_position, display_width, display_height
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-time stereo capture with MatchStereo")
    parser.add_argument('--cap_dir', default='video_cap', type=str, help='Directory to save captured images')
    parser.add_argument('--checkpoint_path', required=False, type=str, help='Path to MatchStereo checkpoint')
    parser.add_argument('--device_id', default=0, type=int, help='Device id of gpu, -1 for cpu')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+', help='Inference size (H W)')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp16')
    parser.add_argument('--variant', choices=['tiny', 'small', 'base'], default='tiny')
    parser.add_argument('--no_compile', action='store_true', default=False, help='Disable torch.compile')
    parser.add_argument('--low_res_init', action='store_true', default=False, help='Use low-resolution init for high-res images')
    parser.add_argument('--mat_impl', choices=['pytorch', 'cuda'], default='cuda', help='MatchAttention implementation')
    parser.add_argument('--mode', choices=['stereo', 'flow'], default='stereo', help='Support stereo and flow tasks')
    parser.add_argument('--zed_res', choices=['1080P', '720P'], default='720P')
    parser.add_argument('--no_disparity', action='store_true', default=False, help='Skip disparity computation, only show rectified images')
    
    args = parser.parse_args()
    
    # Check if checkpoint is required
    if not args.no_disparity and args.checkpoint_path is None:
        parser.error("--checkpoint_path is required when disparity computation is enabled")
    
    # Create save directory
    if not os.path.exists(args.cap_dir):
        os.makedirs(args.cap_dir)
    
    # Create subdirectories in save directory
    sub_dirs = ["left", "right", "left_rectified", "right_rectified", "disparity"]
    if not args.no_disparity:
        sub_dirs.append("mouse_coords")
    
    for dir_name in sub_dirs:
        dir_path = os.path.join(args.cap_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Initialize MatchStereo model only if disparity computation is enabled
    disparity_computer = None
    if not args.no_disparity:
        print("Initializing MatchStereo model...")
        disparity_computer = MatchStereoDisparity(args)
    
    # Initialize camera
    zed = sl.Camera()
    
    init_params = sl.InitParameters()
    if args.zed_res == '720P':
        init_params.camera_resolution = sl.RESOLUTION.HD720
    elif args.zed_res == '1080P':
        init_params.camera_resolution = sl.RESOLUTION.HD1080
    else:
        raise NotImplementedError
    init_params.camera_fps = 30
    err = zed.open(init_params)

    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera open : "+repr(err)+". Exit program.")
        exit()
    
    camera_info = zed.get_camera_information().camera_configuration
    width = camera_info.resolution.width
    height = camera_info.resolution.height
    baseline_mm = camera_info.calibration_parameters.get_camera_baseline()
    f_pixel = camera_info.calibration_parameters.left_cam.fx

    print(f"Focal length: {f_pixel:.2f} pixels")
    print(f"Baseline: {baseline_mm:.2f} mm")
    if args.no_disparity:
        print("Running in NO-DISPARITY mode")
    
    # Set mouse callback for disparity map window
    if not args.no_disparity:
        cv2.namedWindow("Disparity Map")
        cv2.setMouseCallback("Disparity Map", mouse_callback)

    # Set mouse callback for Left Rectified window
    cv2.namedWindow("Left Rectified")
    cv2.setMouseCallback("Left Rectified", left_rectified_mouse_callback)
    
    disparity_visualizer = DisparityVisualizer()
    
    frame = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    i = 1
    try:
        while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            
                zed.retrieve_image(frame, sl.VIEW.SIDE_BY_SIDE, sl.MEM.CPU)
                frame_np = frame.get_data()

                # Split to left and right
                left_frame = frame_np[:, :width, :]
                right_frame = frame_np[:, width:, :]
                
                # RGBA to RGB
                left_rectified, right_rectified = left_frame[..., :3], right_frame[..., :3]
                
                # Compute disparity only if enabled
                if not args.no_disparity:
                    start_time = time.time()
                    disparity = disparity_computer.compute_disparity(left_rectified, right_rectified)
                    last_inference_time = time.time() - start_time
                    fps = 1.0 / last_inference_time if last_inference_time > 0 else 0
                    
                    # Visualize disparity map
                    disparity_colored, disparity_filtered = disparity_visualizer.visualize_disparity_map(disparity)
                    
                    # Update global disparity map variables
                    current_disparity_map = disparity_filtered
                    current_disparity_colored = disparity_colored
                
                # Display rectified images
                left_display = cv2.resize(left_rectified, (display_width, display_height))
                right_display = cv2.resize(right_rectified, (display_width, display_height))
                
                # Draw crosshair on left display if mouse is pressed
                if left_rectified_mouse_pressed:
                    x, y = left_rectified_mouse_position
                    cv2.line(left_display, (x-10, y), (x+10, y), (255, 255, 255), 2)
                    cv2.line(left_display, (x, y-10), (x, y+10), (255, 255, 255), 2)
                    
                cv2.imshow("Left Rectified", left_display)
                cv2.imshow("Right Rectified", right_display)
                
                # Update disparity display only if disparity computation is enabled
                if not args.no_disparity:
                    update_disparity_display(width, height)
                
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save rectified images
                    left_rect_filename = f"left_rectified{i:04d}.png"
                    right_rect_filename = f"right_rectified{i:04d}.png"
                    cv2.imwrite(os.path.join(args.cap_dir, "left_rectified", left_rect_filename), left_rectified)
                    cv2.imwrite(os.path.join(args.cap_dir, "right_rectified", right_rect_filename), right_rectified)
                    
                    if not args.no_disparity:
                        # Save disparity map
                        disparity_filename = f"disparity{i:04d}.tiff"
                        cv2.imwrite(os.path.join(args.cap_dir, "disparity", disparity_filename), disparity)
                        
                        print(f"Images and disparity map saved to {args.cap_dir} directory!")
                        print(f"Rectified left: {left_rect_filename}, Rectified right: {right_rect_filename}")
                        print(f"Disparity map: {disparity_filename}")
                        
                        # Print some disparity statistics
                        valid_disparities = disparity_filtered[~np.isnan(disparity_filtered)]
                        if len(valid_disparities) > 0:
                            print(f"Disparity stats - Min: {np.min(valid_disparities):.2f}px, Max: {np.max(valid_disparities):.2f}px, Mean: {np.mean(valid_disparities):.2f}px")
                    else:
                        print(f"Images saved to {args.cap_dir} directory!")
                        print(f"Rectified left: {left_rect_filename}, Rectified right: {right_rect_filename}")
                    
                    i += 1

    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        zed.close()

if __name__ == "__main__":
    main()