import argparse
import torch
import onnx
from onnxconverter_common import float16

def export_to_onnx(weights_path, img_size=640, output_name="yolop.onnx"):
    print(f"[INFO] Loading model from {weights_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model exactly as your detect.py does
    # We load it to CPU/Float32 for the safest export
    try:
        model = torch.jit.load(weights_path, map_location='cpu')
        model.eval()
        print("[INFO] Model loaded successfully (TorchScript).")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Create dummy input (Batch Size 1, 3 Channels, Height, Width)
    dummy_input = torch.zeros(1, 3, img_size, img_size).to('cpu')

    print(f"[INFO] Exporting to ONNX (FP32) -> {output_name} ...")
    
    # Input and Output names are critical for inference later
    input_names = ['images']
    # YOLOP usually outputs: [det_preds, drive_area_seg, lane_line_seg]
    # Note: Because it is a JIT model, the output structure is fixed by the trace.
    output_names = ['det_out', 'drive_area_seg', 'lane_line_seg']

    torch.onnx.export(
        model, 
        dummy_input, 
        output_name, 
        opset_version=12,  # Opset 12 is usually stable for YOLO-like architectures
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True
    )
    print("[INFO] FP32 Export complete.")

    # --- Convert to FP16 ---
    print("[INFO] Converting ONNX model to FP16...")
    model_fp32 = onnx.load(output_name)
    model_fp16 = float16.convert_float_to_float16(model_fp32)
    
    output_fp16_name = output_name.replace(".onnx", "_fp16.onnx")
    onnx.save(model_fp16, output_fp16_name)
    
    print(f"[SUCCESS] Saved FP16 model to: {output_fp16_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='data/weights/yolopv2.pt', help='path to weights file')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()

    export_to_onnx(opt.weights, opt.img_size)