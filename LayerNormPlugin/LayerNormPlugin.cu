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
#include <cub/cub.cuh>
#include "cublas_v2.h"

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

template<int VPT>
struct BytesToType;

template<>
struct BytesToType<2>
{
    using type = uint16_t;
};
template<>
struct BytesToType<4>
{
    using type = uint32_t;
};
template<>
struct BytesToType<8>
{
    using type = uint64_t;
};
template<>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void *local, void *data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

__global__ void layerNormKernelBasic(float *pInput, float *pGamma, float *pBeta, float *pOutput)
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

template<typename T>
__global__ void layerNormKernelCUBV1(T* pInput, float* pGamma, float* pBeta, T* pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * N_EMBEDDING + tx;
    T _x = pInput[index], _b = (T)pGamma[tx], _a = (T)pBeta[tx];

    __shared__ T mean_shared, var_shared;

    typedef cub::BlockReduce<T, N_EMBEDDING> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    T& ref0 = _x;
    T sum = BlockReduce(temp).Sum(ref0);
    // __syncthreads();
    if (tx == 0)
        mean_shared = sum / T(N_EMBEDDING);
    __syncthreads();

    T moment = _x - mean_shared, moment2 = moment * moment;
    T& ref1 = moment2;
    T var = BlockReduce(temp).Sum(ref1);
    // __syncthreads();
    if (tx == 0)
        var_shared = var / T(N_EMBEDDING);
    __syncthreads();

    pOutput[index] = (moment) * (T)rsqrtf(var_shared + 6e-6) * _b + _a;
}

struct Float2Sum
{
    __device__ __forceinline__ float2 operator()(const float2 &a, const float2 &b) const
    {
        return make_float2(a.x + b.x, a.y + b.y);
    }
};

template <typename T, int TPB, int VPT>
__global__ void layerNormKernelCUBV2(const T* input, const T* gamma, const T* beta, T* output)
{
    const int idx  = blockIdx.x * N_EMBEDDING + threadIdx.x * VPT;
    T localX[VPT], localGamma[VPT], localBeta[VPT];

    copy<sizeof(T) * VPT>(&input[idx], localX);
    float2 localFloat2 = {0.f, 0.f};
    const float rld = float(1) / float(N_EMBEDDING);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const float tmp = rld * (float)localX[it];
        localFloat2.x += tmp;
        localFloat2.y += tmp * (float)localX[it];
    }

    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);
    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);
    using BlockReduce = cub::BlockReduce<float2, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float mu;
    __shared__ float rsigma;
    const float2 sumKV = BlockReduce(temp_storage).Reduce(localFloat2, Float2Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.x;
        rsigma = rsqrtf(sumKV.y - mu * mu + 1e-6);
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        localX[it] = (float)localGamma[it] * ((float)localX[it] - mu) * rsigma + (float)localBeta[it];
    }

    copy<sizeof(T) * VPT>(localX, &output[idx]);
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];

    // layerNormKernelBasic <<<nBlock, N_EMBEDDING / 2, 0, stream>>>((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)outputs[0]);
    // (layerNormKernelCUBV1<float>)<<<nBlock, N_EMBEDDING, 0, stream>>>((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)outputs[0]);
    constexpr int VPT = 16 / sizeof(float);
    constexpr int TPB = N_EMBEDDING / VPT;
    (layerNormKernelCUBV2<float, TPB, VPT>)<<<nBlock, TPB, 0, stream>>>((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)outputs[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);