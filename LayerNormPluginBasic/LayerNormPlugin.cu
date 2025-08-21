/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
 #include "LayerNormPlugin.h"

using namespace nvinfer1;

constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 768 / 2;
constexpr int N_EMBEDDING = 768;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    val = warp_reduce_sum(val);
    return val;
}

__global__ void layerNormKernel(float *pInput, float *pGamma, float *pBeta, float *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * N_EMBEDDING + threadIdx.x;

    __shared__ float temp[N_EMBEDDING / 2];

    float value0 = pInput[index];
    float value1 = pInput[index + N_EMBEDDING / 2];

    temp[tx] = value0 + value1;
    __syncthreads();

    float mean = block_reduce_sum(temp[tx]) / N_EMBEDDING;
    __syncthreads();

    temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean);
    __syncthreads();

    float var = block_reduce_sum(temp[tx]) / N_EMBEDDING;

    int offset = index % N_EMBEDDING;
    float gamma_val0 = pGamma[offset];
    float beta_val0 = pBeta[offset];
    float gamma_val1 = pGamma[offset + N_EMBEDDING / 2];
    float beta_val1 = pBeta[offset + N_EMBEDDING / 2];

    pOutput[index]       = ((value0 - mean) * rsqrtf(var + 6e-6)) * gamma_val0 + beta_val0;
    pOutput[index + N_EMBEDDING / 2] = ((value1 - mean) * rsqrtf(var + 6e-6)) * gamma_val1 + beta_val1;
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];

    layerNormKernel <<<nBlock, N_EMBEDDING / 2, 0, stream>>>((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)outputs[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);