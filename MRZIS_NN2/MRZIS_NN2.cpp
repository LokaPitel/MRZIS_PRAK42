#include <iostream>
#include <string>
#include <iomanip>

#include <cmath>

#include <vector>

std::string getReprOfVector(std::vector<double> vec)
{
	std::string result = "";

	for (int i = 0; i < vec.size(); i++)
		result += std::to_string(vec[i]) + std::string(" ");

	return result;
}

double sigm(double x)
{
	return 1.0 / (1 + std::exp(-x));
}

double sigm_deriv(double x)
{
	return sigm(x) * (1 - sigm(x));
}

double linear_error(double target, double value)
{
	return target - value;
}

class NeuralNetwork
{
	std::vector<double> firstNeuronWeights;
	std::vector<double> secondNeuronWeights;
	std::vector<double> outputNeuronWeights;

	double firstNeuronSum;
	double secondNeuronSum;
	double outputNeuronSum;

	double firstNeuronValue;
	double secondNeuronValue;

	double firstNeuronError;
	double secondNeuronError;

	std::vector<double> lastFirstNeuronDelta;
	std::vector<double> lastSecondNeuronDelta;
	std::vector<double> lastOutputNeuronDelta;

	const double targetValue = 0.9;
	const double alpha = 0.25;

	const int firstLayerNeuronsCount = 2;

public:
	NeuralNetwork()
	{
		firstNeuronWeights.push_back(-0.1); // Threshold
		firstNeuronWeights.push_back(-0.2);
		firstNeuronWeights.push_back(0.1);

		secondNeuronWeights.push_back(-0.1); // Threshold
		secondNeuronWeights.push_back(-0.1);
		secondNeuronWeights.push_back(0.3);

		outputNeuronWeights.push_back(-0.2);
		outputNeuronWeights.push_back(0.2);
		outputNeuronWeights.push_back(0.3);

		double firstNeuronSum = 0;
		double secondNeuronSum = 0;
		double outputNeuronSum = 0;

		double firstNeuronValue = 0;
		double secondNeuronValue = 0;

		firstNeuronError = 0;
		secondNeuronError = 0;
	}

	double forward(std::vector<double> inputVector)
	{
		firstNeuronSum = 0;

		firstNeuronSum -= firstNeuronWeights[0];
		for (int i = 0; i < inputVector.size(); i++)
		{
			firstNeuronSum += firstNeuronWeights[i + 1] * inputVector[i];
		}

		firstNeuronValue = sigm(firstNeuronSum);

		secondNeuronSum = 0;

		secondNeuronSum -= secondNeuronWeights[0];
		for (int i = 0; i < inputVector.size(); i++)
		{
			secondNeuronSum += secondNeuronWeights[i + 1] * inputVector[i];
		}

		secondNeuronValue = sigm(secondNeuronSum);

		std::vector<double> outputNeuronInput;
		outputNeuronInput.push_back(firstNeuronValue);
		outputNeuronInput.push_back(secondNeuronValue);

		outputNeuronSum = 0;

		outputNeuronSum -= outputNeuronWeights[0];
		for (int i = 0; i < outputNeuronInput.size(); i++)
			outputNeuronSum += outputNeuronWeights[i + 1] * outputNeuronInput[i];

		return sigm(outputNeuronSum);
	}

	void nextTrainingIteration(std::vector<double> inputVector)
	{
		double outputNeuronValue = forward(inputVector);

		double outputNeuronFirstWeightDelta = -alpha * linear_error(targetValue, outputNeuronValue) *
			sigm_deriv(outputNeuronSum) * firstNeuronValue;

		double outputNeuronSecondWeightDelta = -alpha * linear_error(targetValue, outputNeuronValue) *
			sigm_deriv(outputNeuronSum) * secondNeuronValue;

		double outputNeuronThresholdDelta = -alpha * linear_error(targetValue, outputNeuronValue) *
			sigm_deriv(outputNeuronSum);

		double firstNeuronFirstWeightDelta = -alpha * linear_error(targetValue, outputNeuronValue) *
			sigm_deriv(outputNeuronSum) * outputNeuronWeights[1] * sigm_deriv(firstNeuronSum) * inputVector[0];

		double firstNeuronSecondWeightDelta = -alpha * linear_error(targetValue, outputNeuronValue) *
			sigm_deriv(outputNeuronSum) * outputNeuronWeights[1] * sigm_deriv(firstNeuronSum) * inputVector[1];

		double firstNeuronThresholdDelta = -alpha * linear_error(targetValue, outputNeuronValue) *
			sigm_deriv(outputNeuronSum) * outputNeuronWeights[1] * sigm_deriv(firstNeuronSum);

		double secondNeuronFirstWeightDelta = -alpha * linear_error(targetValue, outputNeuronValue) *
			sigm_deriv(outputNeuronSum) * outputNeuronWeights[2] * sigm_deriv(secondNeuronSum) * inputVector[0];

		double secondNeuronSecondWeightDelta = -alpha * linear_error(targetValue, outputNeuronValue) *
			sigm_deriv(outputNeuronSum) * outputNeuronWeights[2] * sigm_deriv(secondNeuronSum) * inputVector[1];

		double secondNeuronThresholdDelta = -alpha * linear_error(targetValue, outputNeuronValue) *
			sigm_deriv(outputNeuronSum) * outputNeuronWeights[2] * sigm_deriv(secondNeuronSum);

		firstNeuronError = linear_error(targetValue, outputNeuronValue) * sigm_deriv(outputNeuronSum) * outputNeuronWeights[1];
		secondNeuronError = linear_error(targetValue, outputNeuronValue) * sigm_deriv(outputNeuronSum) * outputNeuronWeights[2];

		lastFirstNeuronDelta.clear();
		lastFirstNeuronDelta.push_back(firstNeuronThresholdDelta);
		lastFirstNeuronDelta.push_back(firstNeuronFirstWeightDelta);
		lastFirstNeuronDelta.push_back(firstNeuronSecondWeightDelta);

		lastSecondNeuronDelta.clear();
		lastSecondNeuronDelta.push_back(secondNeuronThresholdDelta);
		lastSecondNeuronDelta.push_back(secondNeuronFirstWeightDelta);
		lastSecondNeuronDelta.push_back(secondNeuronSecondWeightDelta);

		lastOutputNeuronDelta.clear();
		lastOutputNeuronDelta.push_back(outputNeuronThresholdDelta);
		lastOutputNeuronDelta.push_back(outputNeuronFirstWeightDelta);
		lastOutputNeuronDelta.push_back(outputNeuronSecondWeightDelta);

		outputNeuronWeights[0] += outputNeuronThresholdDelta;
		outputNeuronWeights[1] += outputNeuronFirstWeightDelta;
		outputNeuronWeights[2] += outputNeuronSecondWeightDelta;

		secondNeuronWeights[0] += secondNeuronThresholdDelta;
		secondNeuronWeights[1] += secondNeuronFirstWeightDelta;
		secondNeuronWeights[2] += secondNeuronSecondWeightDelta;

		firstNeuronWeights[0] += firstNeuronThresholdDelta;
		firstNeuronWeights[1] += firstNeuronFirstWeightDelta;
		firstNeuronWeights[2] += firstNeuronSecondWeightDelta;
	}

	double getError(std::vector<double> inputVector)
	{
		return linear_error(targetValue, forward(inputVector));
	}

	std::vector<double> getFirstNeuron() const { return firstNeuronWeights; }
	std::vector<double> getSecondNeuron() const { return secondNeuronWeights; }
	std::vector<double> getOutputNeuron() const { return outputNeuronWeights; }

	std::vector<double> getFirstNeuronDelta() const { return lastFirstNeuronDelta; }
	std::vector<double> getSecondNeuronDelta() const { return lastSecondNeuronDelta; }
	std::vector<double> getOutputNeuronDelta() const { return lastOutputNeuronDelta; }

	double getFirstError() const { return firstNeuronError; }
	double getSecondError() const { return secondNeuronError; }
};

int main()
{
	std::vector<double> inputVector = { 0.1, 0.9 };

	NeuralNetwork network;

	int counter = 0;
	while (network.getError(inputVector) > 0.001)
	{
		network.nextTrainingIteration(inputVector);

		std::cout << "ITERATION #" << ++counter << " ERROR: " << network.getError(inputVector) << "\n";
	}

	std::cout << "OUTPUT: " << network.forward(inputVector) << "\n";

	std::cout << "T" << " #1" << " #2\n";
	std::cout << getReprOfVector(network.getFirstNeuronDelta()) << "\n";
	std::cout << getReprOfVector(network.getSecondNeuronDelta()) << "\n";
	std::cout << getReprOfVector(network.getOutputNeuronDelta()) << "\n\n";

	std::cout << network.getFirstError() << "\n";
	std::cout << network.getSecondError() << "\n";

	return 0;
}