import argparse
import torch
import onnx
from onnxconverter_common import float16

class CleanYOLOP(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x):
        # 1. Run the JIT model
        # Returns: ([pred_list, anchor_grid], seg, ll)
        outputs = self.model(x)
        
        # 2. Unpack the main tuple
        pred_list = outputs[0][0] # The list of 3 detection tensors
        # outputs[0][1] is anchor_grid -> WE IGNORE THIS
        drive_area = outputs[1]
        lane_line = outputs[2]
        
        # 3. Unpack the detection list (The "Hidden" step)
        det_small = pred_list[0]
        det_medium = pred_list[1]
        det_large = pred_list[2]
        
        # 4. Return 5 clean tensors
        return det_small, det_medium, det_large, drive_area, lane_line

def fix_and_export(weights_path, img_size=640):
    print(f"[INFO] Loading JIT model from {weights_path}...")
    jit_model = torch.jit.load(weights_path, map_location='cpu')
    jit_model.eval()
    
    # Wrap it
    model = CleanYOLOP(jit_model)
    print("[INFO] Model wrapped. Structure: 3 Det Heads + 2 Seg Heads.")

    onnx_path_fp32 = weights_path.replace('.pt', '.onnx')
    onnx_path_fp16 = weights_path.replace('.pt', '_fp16.onnx')
    dummy_input = torch.zeros(1, 3, img_size, img_size)

    print("[INFO] Exporting to ONNX (FP32)...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path_fp32,
        opset_version=12,
        input_names=['images'],
        # Naming the 5 outputs clearly for TensorRT
        output_names=['det_small', 'det_medium', 'det_large', 'drive_area_seg', 'lane_line_seg'],
        do_constant_folding=True
    )

    print("[INFO] Converting to FP16...")
    model_fp32 = onnx.load(onnx_path_fp32)
    model_fp16 = float16.convert_float_to_float16(model_fp32)
    onnx.save(model_fp16, onnx_path_fp16)

    print(f"[SUCCESS] Export complete: {onnx_path_fp16}")
    print("Use this file for TensorRT.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='path to yolopv2.pt')
    opt = parser.parse_args()
    fix_and_export(opt.weights)