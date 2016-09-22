#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <random>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


// FFN only holds the configuration of the layer, not actual weights
class FullyConnectedLayer {
    public:
        int inputs, outputs;
        std::vector<float> neurons, bias;

        FullyConnectedLayer(int inputs, int outputs);
};

class TrainingContext {
    public:
        std::default_random_engine rd;

        int batch_size;
        int channels, height, width;

        int train_size;
        int test_size;
        std::vector<float> train_images, train_labels;
        std::vector<float> test_images, test_labels;

        // Disable copying
        TrainingContext& operator=(const TrainingContext&) = delete;
        TrainingContext(const TrainingContext&) = delete;

        // FIXME Fully connected layers should be wrapped into something more manageable
        TrainingContext(int batch_size, FullyConnectedLayer& fc1, FullyConnectedLayer& fc2, FullyConnectedLayer& fc3, FullyConnectedLayer& fc4, std::default_random_engine rd, int train_size, int test_size);
        ~TrainingContext();

        void weight_initialization();
        void initialize(int channels, int height, int width);
        void destroy();

        void train(int epoch);
        void test();

        void forward();
        void backward();
        void update(float learning_rate);

    private:
        cudnnHandle_t cudnn_handle;
        cublasHandle_t cublas_handle;

        // BNN paper says they used 3 hidden layers of 4096 binary units and L2-SVM output layer
        // Dropout was used to regularize the model
        // Square hinge loss (hen L2-SVM) is minimized using ADAM
        // Batch normalization with batch size 100 was used
        // Trained for 1000 epochs for finding the weights with the best validation error

        cudnnTensorDescriptor_t data_tensor, fc1_tensor, fc2_tensor;
        cudnnTensorDescriptor_t fc3_tensor, fc4_tensor;
        cudnnActivationDescriptor_t fc1_activation;
        cudnnActivationDescriptor_t fc2_activation;
        cudnnActivationDescriptor_t fc3_activation;
        FullyConnectedLayer &fc1, &fc2, &fc3, &fc4;

        // Propagation data memory allocation
        float *d_data, *d_labels, *d_fc1_pre, *d_fc1_post, *d_fc2_pre, *d_fc2_post;
        float *d_fc3_pre, *d_fc3_post, *d_fc4_pre, *d_fc4_post;
        // Network parameter memory allocation
        float *w_fc1, *w_fc1bias, *w_fc2, *w_fc2bias;
        float *w_fc3, *w_fc3bias, *w_fc4, *w_fc4bias;
        // Temporary data
        float *d_onevec;

        // Differentials w.r.t. data
        float *dd_fc1, *dd_fc1_post, *dd_fc2, *dd_fc2_post, *d_loss;
        float *dd_fc3, *dd_fc3_post, *dd_fc4, *dd_fc4_post;
        // Network parameter gradients
        float *dw_fc1, *dw_fc1bias, *dw_fc2, *dw_fc2bias;
        float *dw_fc3, *dw_fc3bias, *dw_fc4, *dw_fc4bias;
};
