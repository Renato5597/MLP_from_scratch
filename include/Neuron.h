
#ifndef ML_MLP_NEURON_H
#define ML_MLP_NEURON_H


#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <memory>


double GlorotUniformInitializer(size_t previousLayerSize, size_t currentLayerSize);
double RandomUniformInitializer(size_t previousLayerSize, size_t currentLayerSize);

struct Neuron {
    std::vector<double> weights;
    std::vector<double> m; // for Adam optimizer
    std::vector<double> v; // for Adam optimizer
    double output {};
    double delta {};

    explicit Neuron(size_t numInputs);
};

struct DenseNeurons : public Neuron {
    DenseNeurons(size_t numInputs, size_t previousLayerSize, size_t currentLayerSize);
    void initializeWeightsDense(size_t previousLayerSize, size_t currentLayerSize);
};

struct LinearNeurons : public Neuron {
    LinearNeurons(size_t numInputs, size_t previousLayerSize, size_t currentLayerSize);
    void initializeWeightsLinear(size_t previousLayerSize, size_t currentLayerSize);
};


#endif //ML_MLP_NEURON_H
