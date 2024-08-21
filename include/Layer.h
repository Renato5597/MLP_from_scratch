
#ifndef ML_MLP_LAYER_H
#define ML_MLP_LAYER_H

#include "../include/Neuron.h"

struct Layer
{
public:
    size_t currentLayerSize;
    std::vector<std::unique_ptr<Neuron> > neurons;
};

struct Dense : public Layer
{
public:
    Dense(size_t, size_t, size_t);
};

struct Linear : public Layer
{
public:
    Linear(size_t, size_t, size_t);
};



#endif //ML_MLP_LAYER_H
