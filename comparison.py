import json
import numpy as np
import base64
import io

f = open('onnx_out.pkl','rb')
info_onnx = json.load(f)
f = open('trt_out.pkl','rb')
info_trt = json.load(f)
runners_trt = info_trt["lst"][0][1][0]["outputs"]
runners_onnx = info_onnx["lst"][0][1][0]["outputs"]

for key, value in runners_onnx.items():
    b64_str = value["values"]["array"]
    raw = base64.b64decode(b64_str, validate=True)
    arr = np.load(io.BytesIO(raw), allow_pickle=True)
    if key == "logits":
        print(key, arr)
print('--------------------------')
for key, value in runners_trt.items():
    b64_str = value["values"]["array"]
    raw = base64.b64decode(b64_str, validate=True)
    arr = np.load(io.BytesIO(raw), allow_pickle=True)
    if key == "(Unnamed Layer* 670) [ElementWise]_output":
        print(key, arr)