//
// Created by 39366 on 11/02/2024.
//

#include "../include/Utils.h"

double Mean(const std::vector<double>& vector)
{
    return std::accumulate(vector.begin(), vector.end(), 0.0) / (double)vector.size();
}


double TSS(const std::vector<double>& vector_actual) //total sum of squares
{
    double total = 0.0;
    double y_mean = Mean(vector_actual);
    for(double i : vector_actual)
    {
        double differences = i - y_mean;
        total += differences * differences;
    }
    return total;
}
double RSS(const std::vector<double>& vector_actual, const std::vector<double>& vector_pred) //residual sum of squares
{
    double total = 0.0;
    double residual;
    for(size_t i = 0; i < vector_actual.size(); i++)
    {
        residual = (vector_actual.at(i)-vector_pred.at(i));
        total += residual*residual;
    }
    return total;
}

double r2_score(const std::vector<double>& vector_actual, const std::vector<double>& vector_pred)
{
    double residual_sum_of_squares = RSS(vector_actual, vector_pred);
    double total_sum_of_squares = TSS(vector_actual);
    double r2 = 1 - (residual_sum_of_squares/total_sum_of_squares);
    return r2;
}

double MAE_score(const std::vector<double>& vector_actual, const std::vector<double>& vector_pred)
{
    double total = 0.0;
    for(size_t i = 0; i < vector_actual.size(); i++)
    {
        total += std::abs(vector_actual[i] - vector_pred[i]);
    }
    return total/(double)vector_actual.size();
}

double MSE_score(const std::vector<double>& vector_actual, const std::vector<double>& vector_pred)
{
    double total = 0.0;
    double residual;
    for(size_t i = 0; i < vector_actual.size(); i++)
    {
        residual = (vector_actual.at(i)-vector_pred.at(i));
        total += residual * residual;
    }
    return total/(double)vector_actual.size();
}

