import onnxruntime as ort
for p in ["models/vehicle.onnx", "models/platenum_closeup.onnx"]:
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    meta = s.get_modelmeta().custom_metadata_map
    print(p, "-> input", s.get_inputs()[0].shape, "| classes:", meta.get("names", "?")[:70])
