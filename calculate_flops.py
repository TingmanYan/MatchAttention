import argparse
import torch
from models.match_stereo import MatchStereo

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

def prepare_input(resolution, device, stereo):
    B, C, H, W = resolution
    img0 = torch.FloatTensor(B, C, H, W).to(device)
    img1 = torch.FloatTensor(B, C, H, W).to(device)
    return dict(img0=img0, img1=img1, stereo=stereo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=0, type=int, help='Devide id of gpu, -1 for cpu')
    parser.add_argument('--mode', choices=['stereo', 'flow'], default='stereo', help='Support stereo and flow tasks')
    parser.add_argument('--mat_impl', choices=['pytorch', 'cuda'], default='pytorch', help='MatchAttention implementation')
    parser.add_argument('--variant', choices=['tiny', 'small', 'base'], default='tiny')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+', help='Shall be divisible by 32')
    
    args = parser.parse_args()
    device_id = args.device_id

    if device_id >=0:
        get_accelerator().set_device(device_id)
        device = get_accelerator().device_name(device_id)
    else:
        device = torch.device('cpu')
    model = MatchStereo(args).to(device)
    model.eval()

    input_res=(1, 3, args.inference_size[0], args.inference_size[1]) 
    ## input_res=(1, 3, 2176, 3840) # 4K UHD res, divisible by 32
    ## input_res=(1, 3, 1088, 1920) # 1080P FHD res
    ## input_res=(1, 3, 1536, 1536) # Depth Pro res
    ## input_res=(1, 3, 384, 1248) # KITTI res
    ## input_res=(1, 3, 512, 512)
    with torch.no_grad():
        flops, macs, params = get_model_profile(model=model, 
                                                kwargs=prepare_input(input_res, device, (args.mode=='stereo')), warm_up=0, as_string=True, print_profile=True, detailed=True)
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print(' -Input Res : ', input_res)
    print(' -Flops : ' + flops)
    print(' -Macs : ' + macs)
    print(' -Params: ' + params)