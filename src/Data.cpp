//
// Created by 39366 on 10/02/2024.
//

#include <utility>

#include "../include/Data.h"

//setters
void Data::setFeatureDescriptors(std::vector<std::string> vect)
{
    featureDescriptors = std::move(vect);
}

void Data::setTargetDescriptor(std::string val)
{
    targetDescriptor = std::move(val);
}

void Data::setFeatureVector(std::vector<double> vect)
{
    featureVector = std::move(vect);
}

void Data::setTarget(double val) noexcept
{
    target = val;
}

// append and size
void Data::appendToFeatureDescriptors(const std::string& val)
{
    featureDescriptors.emplace_back(val);
}
void Data::appendToFeatureVector(double val)
{
    featureVector.emplace_back(val);
}

//getters

const std::vector<std::string>& Data::getFeatureDescriptors() const
{
    return featureDescriptors;
}

const std::string& Data::getTargetDescriptor() const
{
    return targetDescriptor;
}

const std::vector<double>& Data::getFeatureVector() const
{
    return featureVector;
}

size_t Data::getFeatureVectorSize() const noexcept
{
    return featureVector.size();
}

double Data::getTarget() const noexcept
{
    return target;
}

