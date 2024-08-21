#ifndef ML_MLP_NEURALNETWORK_H
#define ML_MLP_NEURALNETWORK_H

#include "../include/DataHandler.h"
#include "../include/Layer.h"
#include "../include/Neuron.h"
#include "../include/Utils.h"
#include "../include/CommonData.h"
#include <iostream>


class NeuralNetwork : public CommonData
{
public:
    std::vector<std::unique_ptr<Layer> > layers;
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;
    double t;
public:
    NeuralNetwork(std::vector<size_t>, size_t, size_t, double);
    ~NeuralNetwork();
    std::vector<double> propagateThroughLayer(std::unique_ptr<Layer>&, const std::vector<double>&, bool);
    double FeedForwardPropagation(const Data&);
    double activation(const std::vector<double>&, const std::vector<double>&);
    double ReLU(double);
    double linear(double);
    double derivativeReLu(double);
    double derivativeLinear(double);
    void backPropagation(const Data&);
    void updateWeights(const Data&);
    double predict(const Data&);
    void train(int);
    void test();
    void validate();

};


#endif //ML_MLP_NEURALNETWORK_H
