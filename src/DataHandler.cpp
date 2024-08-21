#include "../include/DataHandler.h"

DataHandler::DataHandler() noexcept
=default;

DataHandler::~DataHandler() noexcept
=default;

void DataHandler::readCSV(const std::string& path, const char& delimiter)
{
    //Open file
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Error: can't load file at path " + path);
    }
    std::string line;
    std::string cell;
    int line_number = 0;
    while(std::getline(file, line))
    {
        //Read all the cells in the line
        std::stringstream lineStream(line);
        Data d;
        int target_position = 0;
        while (std::getline(lineStream, cell, delimiter))
        {
            if(line_number > 0 and target_position > 0)
            {
                d.appendToFeatureVector(std::stod(cell));
            }
            else if (line_number > 0 and target_position == 0)
            {
                d.setTarget(std::stod(cell));
            }
            else if (line_number == 0 and target_position == 0)
            {
                d.setTargetDescriptor(cell);
            }
            else if(line_number == 0 and target_position > 0)
            {
                d.appendToFeatureDescriptors(cell);
            }
            target_position++;
        }
        line_number++;
        dataArray.emplace_back(d);
    }
    std::cout<< "Successfully load csv file of path " << path.c_str() << std::endl;
}

void DataHandler::TrainTestSplit()
{
    size_t train_size =  (dataArray.size() - 1) * TRAIN_SET / 100;  //avoid descriptor vector/first row
    size_t test_size =  (dataArray.size() - 1) * TEST_SET / 100;
    size_t validation_size =  (dataArray.size() - 1) * VALIDATION_SET / 100;

    // vector of indexes
    std::vector<int> indexes(dataArray.size() - 1);
    std::iota(indexes.begin(), indexes.end(), 1);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // for reproducibility
    /*auto seed = std::default_random_engine {81};*/
    std::shuffle(indexes.begin(), indexes.end(), std::default_random_engine(seed));

    //reserve
    trainingData.reserve(train_size);
    testData.reserve(test_size);
    validationData.reserve(validation_size);

    // new vectors
    int count = 0;
    int index_count = 0;
    while(count < indexes.size())
    {
        if (index_count < train_size)
        {
            trainingData.emplace_back(dataArray.at(indexes[index_count]));
        }
        else if (index_count >= train_size && index_count < train_size + test_size)
        {
            testData.emplace_back(dataArray.at(indexes[index_count]));
        }
        else
        {
            validationData.emplace_back(dataArray.at(indexes[index_count]));
        }
        count++;
        index_count++;
    }
    std::cout <<"Successfully shuffle and split data."<< std::endl;
    std::cout << "-------------------------------------------"<< std::endl;
    std::cout<< "Training dataset size: "<< trainingData.size() << "\n";
    std::cout<< "Test dataset size: "<< testData.size() << "\n";
    std::cout<< "Validation dataset size: "<< validationData.size() <<"\n";
    std::cout << "-------------------------------------------"<< std::endl;
}

//print
void DataHandler::printDataArray() const
{
    std::cout<< "DataArray:\n";
    int position = 0;
    for (const auto& data: dataArray)
    {
        if(position > 0)
        {
            std::cout << "[ " << data.getTarget() << " ] ->" ;
            std::cout << " [ ";
            for (const auto& value: data.getFeatureVector())
            {
                std::cout<< value << " ";
            }
            std::cout << "]\n";
        }else
        {
            std::cout << "[ " << data.getTargetDescriptor() << " ] ->";
            std::cout << " [ ";
            for (const auto& value: data.getFeatureDescriptors())
            {
                std::cout<< value << " ";
            }
            std::cout << "]\n";
        }
        position++;
    }
}

void DataHandler::printTrainingData() const
{
    std::cout<< "TrainingData:\n";
    for (const auto& data: trainingData)
    {
        std::cout << "[ " << data.getTarget() << " ] ->" ;
        std::cout << " [ ";
        for (const auto& value: data.getFeatureVector())
        {
            std::cout<< value << " ";
        }
        std::cout << "]\n";
    }
}

void DataHandler::printTestData() const
{
    std::cout<< "TestData:\n";
    for (const auto& data: testData)
    {
        std::cout << "[ " << data.getTarget() << " ] ->" ;
        std::cout << " [ ";
        for (const auto& value: data.getFeatureVector())
        {
            std::cout<< value << " ";
        }
        std::cout << "]\n";
    }
}

void DataHandler::printValidationData() const
{
    std::cout<< "ValidationData:\n";
    for (const auto& data: validationData)
    {
        std::cout << "[ " << data.getTarget() << " ] ->" ;
        std::cout << " [ ";
        for (const auto& value: data.getFeatureVector())
        {
            std::cout<< value << " ";
        }
        std::cout << "]\n";
    }
}

//getters

const std::vector<Data>& DataHandler::getTrainingData() const
{
    return trainingData;
}

const std::vector<Data>& DataHandler::getTestData() const
{
    return testData;
}

const std::vector<Data>& DataHandler::getValidationData() const
{
    return validationData;
}
