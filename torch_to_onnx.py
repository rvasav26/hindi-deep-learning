import torch
from devanagari_model import DevanagariCNN
from onnxruntime.quantization import quantize_dynamic, QuantType

model = DevanagariCNN()

state_dict = torch.load(
    "hindi_cnn_weights_pytorch.pt",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, 1, 32, 32)

torch.onnx.export(
    model,
    dummy_input,
    "hindi_cnn.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
)

quantize_dynamic(
    model_input="hindi_cnn.onnx",
    model_output="hindi_cnn_int8.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul"],
)
