#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <chrono>

#include <gflags/gflags.h>

#include "readubyte.h"
#include "ffn.h"

DEFINE_int32(iterations, 1000, "Number of iterations for training");
DEFINE_int32(random_seed, -1, "Override random seed (default uses std::random_device)");
DEFINE_int32(batch_size, 641, "Mini-batch size");

DEFINE_string(train_images, "../data/train-images-idx3-ubyte", "Training images filename");
DEFINE_string(train_labels, "../data/train-labels-idx1-ubyte", "Training labels filename");
DEFINE_string(test_images, "../data/t10k-images-idx3-ubyte", "Test images filename");
DEFINE_string(test_labels, "../data/t10k-labels-idx1-ubyte", "Test labels filename");


using namespace std;

void SavePGMFile(const unsigned char *data, size_t width, size_t height, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (fp)
    {
        fprintf(fp, "P5\n%lu %lu\n255\n", width, height);
        fwrite(data, sizeof(unsigned char), width * height, fp);
        fclose(fp);
    }
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Read train/test data
    size_t width, height, channels = 1;
    size_t train_size = ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), nullptr, nullptr, width, height);
    size_t test_size = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), nullptr, nullptr, width, height);

    std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
    std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);

    if (ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), &train_images[0], &train_labels[0], width, height) != train_size)
        return 2;
    if (ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width, height) != test_size)
        return 3;

    // Save a random image
    // std::random_device rd_image;
    // int random_image = rd_image() % train_size;
    // std::stringstream ss; ss << "image-" << (int)train_labels[random_image] << ".pgm";
    // SavePGMFile(&train_images[0] + random_image * width*height*channels, width, height, ("../result/" + ss.str()).c_str());

    // Seed RNG
    std::random_device r;
    std::default_random_engine rdengine(FLAGS_random_seed < -1 ? r() : FLAGS_random_seed);

    // Set up FFN
    int n_hidden_nodes = 500;
    FullyConnectedLayer fc1(width * height, n_hidden_nodes);
    FullyConnectedLayer fc2(fc1.outputs, 10);

    int batch_size = FLAGS_batch_size;

    // Setup training context
    TrainingContext context(batch_size, fc1, fc2, rdengine, train_size, test_size);

    std::cout << "Weight initialization" << std::endl;

    context.weight_initialization();

    std::cout << "Device initialization" << std::endl;

    context.initialize(channels, height, width);

    std::cout << "Image normalization" << std::endl;

    // Image normalization
    context.train_images.resize(train_images.size());
    context.train_labels.resize(train_size);
    context.test_images.resize(test_images.size());
    context.test_labels.resize(test_size);

    std::cout << "Total of " << train_size << " training images" << std::endl;

    for (size_t i = 0; i < train_size * channels * width * height; ++i) {
        context.train_images[i] = (float)train_images[i] / 255.0f;
    }

    for (size_t i = 0; i < train_size; ++i) {
        context.train_labels[i] = (float)train_labels[i];
    }

    for (size_t i = 0; i < test_size * channels * width * height; ++i) {
        context.test_images[i] = (float)test_images[i] / 255.0f;
    }

    for (size_t i = 0; i < test_size; ++i) {
        context.test_labels[i] = (float)test_labels[i];
    }

    std::cout << "Train" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    context.train(FLAGS_iterations);
    auto t2 = std::chrono::high_resolution_clock::now();

    printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);
    printf("Total time: %f min\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / 1000.0f / 60.0f);

    std::cout << "Test" << std::endl;
    context.test();

    context.destroy();

    return 0;
}
