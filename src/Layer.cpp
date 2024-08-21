
#include "../include/Layer.h"


Dense::Dense(size_t numInputs, size_t previousLayerSize, size_t currentLayerSize)
{
    for(size_t i = 0; i < currentLayerSize; i++)
    {
        neurons.push_back(std::make_unique<DenseNeurons>(numInputs, previousLayerSize, currentLayerSize));
    }
    this->currentLayerSize = currentLayerSize;
}

Linear::Linear(size_t numInputs, size_t previousLayerSize, size_t currentLayerSize)
{
    for(size_t i = 0; i < currentLayerSize; i++)
    {
        neurons.push_back(std::make_unique<LinearNeurons>(numInputs, previousLayerSize, currentLayerSize));
    }
    this->currentLayerSize = currentLayerSize;
}