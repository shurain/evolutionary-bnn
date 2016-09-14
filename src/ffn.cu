#include <random>
#include <cmath>
#include <algorithm>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "ffn.h"

// Block width for CUDA kernel
// FIXME This is super naive
#define BW 128

__global__ void FillOnes(float *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    vec[idx] = 1.0f;
}

static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}

__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int label_value = static_cast<int>(label[idx]);

    // For each item in the batch, decrease the result of the label's value by 1
    diff[idx * num_labels + label_value] -= 1.0f;
}

FullyConnectedLayer::FullyConnectedLayer(int inputs, int outputs)
    : inputs(inputs),
    outputs(outputs),
    neurons(inputs * outputs),
    bias(outputs) {}


TrainingContext::TrainingContext(int batch_size, FullyConnectedLayer& fc1, FullyConnectedLayer& fc2, std::default_random_engine rd, int train_size, int test_size)
    : batch_size(batch_size),
    fc1(fc1),
    fc2(fc2),
    rd(rd),
    train_size(train_size),
    test_size(test_size) {
        checkCudaErrors(cudaSetDevice(0));

        checkCudaErrors(cublasCreate(&cublas_handle));
        checkCUDNN(cudnnCreate(&cudnn_handle));

        checkCUDNN(cudnnCreateTensorDescriptor(&data_tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc1_tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc2_tensor));

        checkCUDNN(cudnnCreateActivationDescriptor(&fc1_activation));

        checkCUDNN(cudnnSetTensor4dDescriptor(fc1_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, fc1.outputs, 1, 1));
        checkCUDNN(cudnnSetTensor4dDescriptor(fc2_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, fc2.outputs, 1, 1));
        checkCUDNN(cudnnSetActivationDescriptor(fc1_activation, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    }

TrainingContext::~TrainingContext() {
    checkCudaErrors(cublasDestroy(cublas_handle));
    checkCUDNN(cudnnDestroy(cudnn_handle));

    checkCUDNN(cudnnDestroyTensorDescriptor(data_tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(fc1_tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(fc2_tensor));
    checkCUDNN(cudnnDestroyActivationDescriptor(fc1_activation));
}

void TrainingContext::weight_initialization() {
    // Xavier init
    float wfc1 = 4.0f * sqrt(6.0f / (fc1.inputs + fc1.outputs));
    std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
    float wfc2 = 4.0f * sqrt(6.0f / (fc2.inputs + fc2.outputs));
    std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

    std::generate_n(fc1.neurons.begin(), fc1.neurons.size(), [&]{ return dfc1(rd);});
    std::generate_n(fc2.neurons.begin(), fc2.neurons.size(), [&]{ return dfc2(rd);});

    std::generate_n(fc1.bias.begin(), fc1.bias.size(), [&]{ return dfc1(rd);});
    std::generate_n(fc2.bias.begin(), fc2.bias.size(), [&]{ return dfc2(rd);});
}

void TrainingContext::initialize(int channels, int height, int width) {
    this->channels = channels;
    this->height = height;
    this->width = width;
    // Memory allocation for the propagated data
    checkCudaErrors(cudaMalloc(&d_data, sizeof(float) * this->batch_size * channels * height * width));
    checkCudaErrors(cudaMalloc(&d_labels, sizeof(float) * this->batch_size));
    checkCudaErrors(cudaMalloc(&d_fc1_pre, sizeof(float) * this->batch_size * fc1.outputs));
    checkCudaErrors(cudaMalloc(&d_fc1_post, sizeof(float) * this-> batch_size * fc1.outputs));
    checkCudaErrors(cudaMalloc(&d_fc2_pre, sizeof(float) * this->batch_size * fc2.outputs));
    checkCudaErrors(cudaMalloc(&d_fc2_post, sizeof(float) * this->batch_size * fc2.outputs));

    // Memory allocation for the network parameters
    checkCudaErrors(cudaMalloc(&w_fc1, sizeof(float) * fc1.neurons.size()));
    checkCudaErrors(cudaMalloc(&w_fc1bias, sizeof(float) * fc1.bias.size()));
    checkCudaErrors(cudaMalloc(&w_fc2, sizeof(float) * fc2.neurons.size()));
    checkCudaErrors(cudaMalloc(&w_fc2bias, sizeof(float) * fc2.bias.size()));

    // Memory allocation for temporary data
    checkCudaErrors(cudaMalloc(&d_onevec, sizeof(float) * batch_size));

    // Memory allocation for differentials
    checkCudaErrors(cudaMalloc(&dd_fc1, sizeof(float) * batch_size * fc1.inputs));
    checkCudaErrors(cudaMalloc(&dd_fc1_post, sizeof(float) * batch_size * fc1.outputs));
    checkCudaErrors(cudaMalloc(&dd_fc2, sizeof(float) * batch_size * fc2.inputs));
    checkCudaErrors(cudaMalloc(&dd_fc2_post, sizeof(float) * batch_size * fc2.outputs));
    checkCudaErrors(cudaMalloc(&d_loss, sizeof(float) * batch_size * fc2.outputs));

    // Memory allocation for gradients of network parameters
    checkCudaErrors(cudaMalloc(&dw_fc1, sizeof(float) * fc1.neurons.size()));
    checkCudaErrors(cudaMalloc(&dw_fc1bias, sizeof(float) * fc1.bias.size()));
    checkCudaErrors(cudaMalloc(&dw_fc2, sizeof(float) * fc2.neurons.size()));
    checkCudaErrors(cudaMalloc(&dw_fc2bias, sizeof(float) * fc2.bias.size()));

    // Populate GPU global memory
    checkCudaErrors(cudaMemcpyAsync(w_fc1, &fc1.neurons[0], sizeof(float) * fc1.neurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(w_fc1bias, &fc1.bias[0], sizeof(float) * fc1.bias.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(w_fc2, &fc2.neurons[0], sizeof(float) * fc2.neurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(w_fc2bias, &fc2.bias[0], sizeof(float) * fc2.bias.size(), cudaMemcpyHostToDevice));

    FillOnes<<<RoundUp(batch_size, BW), BW>>>(d_onevec, batch_size);
}

void TrainingContext::forward() {
    float alpha = 1.0f, beta = 0.0f;

    // Input -> layer 1 propagation
    checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, fc1.outputs, batch_size, fc1.inputs, &alpha, w_fc1, fc1.inputs, d_data, fc1.inputs, &beta, d_fc1_pre, fc1.outputs));
    // Add layer 1 bias
    checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, fc1.outputs, batch_size, 1, &alpha, w_fc1bias, fc1.outputs, d_onevec, 1, &alpha, d_fc1_pre, fc1.outputs));
    // Apply sigmoid activation
    checkCUDNN(cudnnActivationForward(cudnn_handle, fc1_activation, &alpha, fc1_tensor, d_fc1_pre, &beta, fc1_tensor, d_fc1_post));

    // Layer 1 -> layer 2 propagation
    checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, fc2.outputs, batch_size, fc2.inputs, &alpha, w_fc2, fc2.inputs, d_fc1_post, fc2.inputs, &beta, d_fc2_pre, fc2.outputs));
    // Add layer 2 bias
    checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, fc2.outputs, batch_size, 1, &alpha, w_fc2bias, fc2.outputs, d_onevec, 1, &alpha, d_fc2_pre, fc2.outputs));
    // Softmax loss
    checkCUDNN(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, fc2_tensor, d_fc2_pre, &beta, fc2_tensor, d_fc2_post));
}

void TrainingContext::backward() {
    float alpha = 1.0f, beta = 0.0f;

    // FIXME I don't really understand what this is
    float scale_value = 1.0f / static_cast<float>(batch_size);

    checkCudaErrors(cudaMemcpyAsync(d_loss, d_fc2_post, sizeof(float) * batch_size * fc2.outputs, cudaMemcpyDeviceToDevice));

    SoftmaxLossBackprop<<<RoundUp(batch_size, BW), BW>>>(d_labels, fc2.outputs, batch_size, d_loss);

    checkCudaErrors(cublasSscal(cublas_handle, fc2.outputs * batch_size, &scale_value, d_loss, 1));

    // Layer 2
    // Derivative w.r.t. weights dw_fc2 = (d_fc1_post * d_fc2_post)
    checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, fc2.inputs, fc2.outputs, batch_size, &alpha, d_fc1_post, fc2.inputs, d_loss, fc2.outputs, &beta, dw_fc2, fc2.inputs));
    // Derivative w.r.t. bias dw_fc2bias = d_loss * 1_vec
    checkCudaErrors(cublasSgemv(cublas_handle, CUBLAS_OP_N, fc2.outputs, batch_size, &alpha, d_loss, fc2.outputs, d_onevec, 1, &beta, dw_fc2bias, 1));
    // Derivative w.r.t. data d_fc2_pre * d_fc2_post
    checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, fc2.inputs, batch_size, fc2.outputs, &alpha, w_fc2, fc2.inputs, d_loss, fc2.outputs, &beta, dd_fc2, fc2.inputs));
    checkCUDNN(cudnnActivationBackward(cudnn_handle, fc1_activation, &alpha, fc1_tensor, d_fc1_post, fc1_tensor, dd_fc2, fc1_tensor, d_fc1_pre, &beta, fc1_tensor, dd_fc1_post));

    // Layer 1
    // Derivative w.r.t. weights d_g_fc1 = d_data * d_fc1_post)
    checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, fc1.inputs, fc1.outputs, batch_size, &alpha, d_data, fc1.inputs, dd_fc1_post, fc1.outputs, &beta, dw_fc1, fc1.inputs));
    // Derivative w.r.t. bias dw_fc1bias = d_fc1_post * 1_vec
    checkCudaErrors(cublasSgemv(cublas_handle, CUBLAS_OP_N, fc1.outputs, batch_size, &alpha, dd_fc1_post, fc1.outputs, d_onevec, 1, &beta, dw_fc1bias, 1));
}

void TrainingContext::update(float learning_rate) {
    float alpha = -learning_rate;

    checkCudaErrors(cublasSaxpy(cublas_handle, static_cast<int>(fc1.neurons.size()), &alpha, dw_fc1, 1, w_fc1, 1));
    checkCudaErrors(cublasSaxpy(cublas_handle, static_cast<int>(fc1.bias.size()), &alpha, dw_fc1bias, 1, w_fc1bias, 1));

    checkCudaErrors(cublasSaxpy(cublas_handle, static_cast<int>(fc2.neurons.size()), &alpha, dw_fc2, 1, w_fc2, 1));
    checkCudaErrors(cublasSaxpy(cublas_handle, static_cast<int>(fc2.bias.size()), &alpha, dw_fc2bias, 1, w_fc2bias, 1));
}

void TrainingContext::train(int iter) {
    checkCudaErrors(cudaDeviceSynchronize());
    for (int i = 0; i < iter; ++i) {
        int imageid = i % (train_size / batch_size);

        // Copy current batch to GPU
        checkCudaErrors(cudaMemcpyAsync(d_data, &train_images[imageid * batch_size * width * height * channels], sizeof(float) * batch_size * channels * width * height, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_labels, &train_labels[imageid * batch_size], sizeof(float) * batch_size, cudaMemcpyHostToDevice));

        forward();

        backward();

        float learning_rate = static_cast<float>(0.01 * pow((1.0 + 0.0001 * i), (-0.75)));
        update(learning_rate);

        if (i % 1000 == 0) {
            std::cout << ".";
            fflush(stdout);
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
}

void TrainingContext::test() {
    checkCudaErrors(cudaDeviceSynchronize());

    int num_errors = 0;

    for (int i = 0; i < test_size; ++i) {
        // We'll test images one by one (instead of mini-batch style)
        std::vector<float> data(width * height);
        for (int j = 0; j < width * height; ++j) {
            data[j] = (float)test_images[i * width * height * channels + j];
        }

        checkCudaErrors(cudaMemcpyAsync(d_data, &data[0], sizeof(float) * channels * width * height, cudaMemcpyHostToDevice));

        forward();

        std::vector<float> class_vector(10);
        checkCudaErrors(cudaMemcpy(&class_vector[0], d_fc2_post, sizeof(float) * 10, cudaMemcpyDeviceToHost));

        int chosen = 0;
        for (int id = 1; id < 10; ++id) {
            if (class_vector[chosen] < class_vector[id]) {
                chosen = id;
            }
        }

        if (chosen != test_labels[i]) {
            num_errors++;
        }
    }
    float total_error = (float) num_errors / (float) test_size;
    printf("Classification result: %.2f%% error (used %d images)\n", total_error * 100.0f, (int) test_size);
}

void TrainingContext::destroy() {
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_labels));
    checkCudaErrors(cudaFree(d_fc1_pre));
    checkCudaErrors(cudaFree(d_fc1_post));
    checkCudaErrors(cudaFree(d_fc2_pre));
    checkCudaErrors(cudaFree(d_fc2_post));

    checkCudaErrors(cudaFree(w_fc1));
    checkCudaErrors(cudaFree(w_fc1bias));
    checkCudaErrors(cudaFree(w_fc2));
    checkCudaErrors(cudaFree(w_fc2bias));

    checkCudaErrors(cudaFree(d_onevec));

    checkCudaErrors(cudaFree(dd_fc1));
    checkCudaErrors(cudaFree(dd_fc1_post));
    checkCudaErrors(cudaFree(dd_fc2));
    checkCudaErrors(cudaFree(dd_fc2_post));
    checkCudaErrors(cudaFree(d_loss));

    checkCudaErrors(cudaFree(dw_fc1));
    checkCudaErrors(cudaFree(dw_fc1bias));
    checkCudaErrors(cudaFree(dw_fc2));
    checkCudaErrors(cudaFree(dw_fc2bias));
}
