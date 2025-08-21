## BERT-TRT
This project provide code for multiple methods of converting onnx BERT model to TensorRT. 

### Usage

To convert BERT model to onnx BERT model, run: 
```shell
python model2onnx.py
```

To convert onnx BERT model to a simplified onnx BERT model, run:
```shell
onnxsim bert-base-uncased/model.onnx bert-base-uncased/model-sim.onnx 
```

To convert a simplified onnx BERT model to TensorRT BERT model, run:
```shell
python onnx2trt.py
```

To convert onnx BERT model to TensorRT BERT model, run: 
```shell
python builder.py -x bert-base-uncased/model.onnx -c bert-base-uncased/ -o bert-base-uncased/model.plan | tee log.txt
```

To compare onnx BERT model with TensorRT BERT model, run:
```shell
polygraphy run bert-base-uncased/model.onnx --onnxrt --data-loader-script onnx_data_loader.py --input-shape input_ids:[1,16] attention_mask:[1,16] token_type_ids:[1,16] --onnx-outputs mark all --save-results=onnx_out.pkl
polygraphy run bert-base-uncased/model.plan --plugins "D:\Software\TensorRT-10.9.0.34\lib\nvinfer_plugin_10.dll" ".\LayerNormPluginBasic\LayerNorm.dll" --trt --data-loader-script trt_data_loader.py --input-shape input_ids:[1,16] token_type_ids:[1,16] position_ids:[1,16] --validate --trt-outputs mark all --save-results=trt_out.pkl
python comparison.py
```