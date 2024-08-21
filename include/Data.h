//
// Created by 39366 on 10/02/2024.
//

#ifndef ML_MLP_DATA_H
#define ML_MLP_DATA_H

#include <iostream>
#include <vector>
#include <string>

class Data
{
private:
     std::vector<std::string> featureDescriptors;
     std::string  targetDescriptor;
     std::vector<double> featureVector;
     double target;
public:
    //setters
    void setFeatureDescriptors(std::vector<std::string>);
    void setTargetDescriptor(std::string);
    void setFeatureVector(std::vector<double>);
    void setTarget(double) noexcept;

    // append and size
    void appendToFeatureDescriptors(const std::string&);
    void appendToFeatureVector(double);

    //getters
    [[nodiscard]] const std::vector<std::string>& getFeatureDescriptors() const;
    [[nodiscard]] const std::string& getTargetDescriptor() const;
    [[nodiscard]] const std::vector<double>& getFeatureVector() const;
    [[nodiscard]] size_t getFeatureVectorSize() const noexcept;
    [[nodiscard]] double getTarget() const noexcept;
};

#endif //ML_MLP_DATA_H
