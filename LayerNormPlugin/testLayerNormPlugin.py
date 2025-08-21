#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import ctypes
import numpy as np
import cuda.bindings.runtime as cudart  # 使用 cuda runtime API
import tensorrt as trt
import platform

if platform.system() == 'Windows':
    soFilePath  = './LayerNorm.dll'
else:
    soFilePath  = './LayerNorm.so'
nBS             = 4
nSL             = 16
nEmbedding      = 768
epsilon         = 6e-6

np.random.seed(97)

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def layerNormCPU(bufferH):
    _x,b,a = bufferH
    nEmbed = bufferH[0].shape[2]
    _0  = np.mean(_x,2)[:,:,np.newaxis]
    _1  = _x - _0
    _2  = _1 * _1
    _3  = np.mean(_2,2)[:,:,np.newaxis]
    _4  = np.array(epsilon,dtype=np.float32)
    _5  = _4.reshape(1,1,1)
    _6  = _3 + _5
    _7  = np.sqrt(_6)
    _8  = 1 / _7                # 1/sqrt(...)
    _9  = b
    _10 = _9.reshape(1,1,nEmbed)
    _11 = _8 * _10              # b/sqrt(...)
    _12 = _0 * _11              # bμ/sqrt(...)
    _13 = a
    _14 = _13.reshape(1,1,nEmbed)
    _15 = _14 - _12             # a-bμ/sqrt(...)
    _16 = _x * _11              # bx/sqrt(...)
    _17 = _15 + _16             # b(x-μ)/sqrt(...)+a
    _18 = _17.reshape(bufferH[0].shape[0],bufferH[0].shape[1],bufferH[0].shape[2])
    return _18

def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        # print(c.name)
        if c.name == 'LayerNorm':
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<0)
    config          = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
    config.flags    = 0

    inputTensorList = []
    inputTensorList.append( network.add_input('inputT', trt.float32, [-1,-1,nEmbedding]) )
    inputTensorList.append( network.add_input('inputB', trt.float32, [nEmbedding]) )
    inputTensorList.append( network.add_input('inputA', trt.float32, [nEmbedding]) )

    profile = builder.create_optimization_profile()
    profile.set_shape('inputT',[nBS,nSL,nEmbedding],[nBS,nSL,nEmbedding],[nBS,nSL,nEmbedding])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())

    network.mark_output(pluginLayer.get_output(0))

    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_input_shape(engine.get_tensor_name(0),[nBS,nSL,nEmbedding])
    context.set_input_shape(engine.get_tensor_name(1),[nEmbedding])
    context.set_input_shape(engine.get_tensor_name(2),[nEmbedding])
    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    
    nInput = np.sum([ engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT for i in range(engine.num_io_tensors) ])
    nOutput = engine.num_io_tensors - nInput
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        print("input ->" if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT else "output->",engine.get_tensor_dtype(tensor_name),engine.get_tensor_shape(tensor_name),context.get_tensor_shape(tensor_name))

    bufferH = []
    bufferH.append( np.random.rand(nBS,nSL,nEmbedding).astype(np.float32).reshape(nBS,nSL,nEmbedding) * 2 - 1)
    bufferH.append( np.ones(nEmbedding).astype(np.float32) )
    bufferH.append( np.zeros(nEmbedding).astype(np.float32) )
    bufferH.append(np.empty(context.get_tensor_shape(engine.get_tensor_name(3)),dtype=trt.nptype(engine.get_tensor_dtype(engine.get_tensor_name(3)))))

    bufferD = []
    for i in range(engine.num_io_tensors):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    noerror = context.execute_v2(bufferD)
    if not noerror:
        raise ValueError("ERROR: inference failed.")

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print("check result:")
    temp1 = bufferH[-1]
    temp2 = layerNormCPU(bufferH[:3])
    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )
    
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == '__main__':
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()