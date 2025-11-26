import argparse
import torch
import onnx
from onnxconverter_common import float16
from onnxsim import simplify  # <--- The Magic Fix

# --- Wrapper Class ---
class CleanYOLOP(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x):
        # 1. Run JIT
        outputs = self.model(x)
        
        # 2. Unpack
        pred_list = outputs[0][0] 
        drive_area = outputs[1]
        lane_line = outputs[2]
        
        # 3. Unpack Detections
        det_small = pred_list[0]
        det_medium = pred_list[1]
        det_large = pred_list[2]
        
        return det_small, det_medium, det_large, drive_area, lane_line

def fix_and_export(weights_path, img_size=640):
    print(f"[INFO] Loading JIT model from {weights_path}...")
    jit_model = torch.jit.load(weights_path, map_location='cpu')
    jit_model.eval()
    
    wrapper = CleanYOLOP(jit_model)
    
    # Scripting the wrapper
    print("[INFO] Scripting model...")
    model = torch.jit.script(wrapper)
    
    onnx_path_fp32 = weights_path.replace('.pt', '.onnx')
    onnx_path_fp16 = weights_path.replace('.pt', '_fp16.onnx') # Final path
    dummy_input = torch.zeros(1, 3, img_size, img_size)

    # --- 1. Export to ONNX (FP32) ---
    print("[INFO] Exporting to ONNX (Opset 11)...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path_fp32,
        opset_version=11,  # <--- CHANGED TO 11 (Better for Jetson)
        input_names=['images'],
        output_names=['det_small', 'det_medium', 'det_large', 'drive_area_seg', 'lane_line_seg'],
        do_constant_folding=True,
        example_outputs=model(dummy_input)
    )

    # --- 2. Run ONNX Simplifier (THE FIX) ---
    print("[INFO] Running ONNX Simplifier (Fixing Resize layers)...")
    onnx_model = onnx.load(onnx_path_fp32)
    model_simp, check = simplify(onnx_model)
    
    if not check:
        print("[ERROR] Simplification failed!")
        return
    else:
        print("[INFO] Simplification successful.")

    # --- 3. Convert Simplified Model to FP16 ---
    print("[INFO] Converting to FP16...")
    model_fp16 = float16.convert_float_to_float16(model_simp)
    onnx.save(model_fp16, onnx_path_fp16)

    print(f"[SUCCESS] optimized model ready for TensorRT: {onnx_path_fp16}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,  default='data/weights/yolopv2.pt', required=True, help='path to yolopv2.pt')
    opt = parser.parse_args()
    fix_and_export(opt.weights)