#include "../include/Neuron.h"

double GlorotUniformInitializer(size_t previousLayerSize, size_t currentLayerSize) {
    std::random_device rd;
    std::uniform_real_distribution<double> distribution(-std::sqrt(6.0 / (double)(previousLayerSize + currentLayerSize)),
                                                        std::sqrt(6.0 / (double)(previousLayerSize + currentLayerSize)));
    return distribution(rd);
}

double RandomUniformInitializer(size_t previousLayerSize, size_t currentLayerSize) {
    std::random_device rd;
    std::uniform_real_distribution<double> distribution(-std::sqrt(1.0 / (double) previousLayerSize),
                                                        std::sqrt(1.0 / (double) previousLayerSize));
    return distribution(rd);
}

Neuron::Neuron(size_t numInputs) {
    weights.resize(numInputs + 1); // +1 for the bias term
    m.resize(numInputs + 1, 0.0);
    v.resize(numInputs + 1, 0.0);
    // Initialize weights, for example with random values
}

DenseNeurons::DenseNeurons(size_t numInputs, size_t previousLayerSize, size_t currentLayerSize) : Neuron(numInputs) {
    initializeWeightsDense(previousLayerSize, currentLayerSize);
}

void DenseNeurons::initializeWeightsDense(size_t previousLayerSize, size_t currentLayerSize) {
    for (size_t i = 0; i < previousLayerSize + 1; ++i) {
        if (i < previousLayerSize) {
            weights[i] = GlorotUniformInitializer(previousLayerSize, currentLayerSize);
        } else {
            weights[i] = 0.0; // bias zero
        }
    }
}

LinearNeurons::LinearNeurons(size_t numInputs, size_t previousLayerSize, size_t currentLayerSize) : Neuron(numInputs) {
    initializeWeightsLinear(previousLayerSize, currentLayerSize);
}

void LinearNeurons::initializeWeightsLinear(size_t previousLayerSize, size_t currentLayerSize) {
    for (size_t i = 0; i < previousLayerSize + 1; ++i) {
        weights[i] = RandomUniformInitializer(previousLayerSize, currentLayerSize);
    }
}
