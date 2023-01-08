package me.vjta088x.ai.network;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;
import me.vjta088x.ai.utils.UtilMath;

import java.io.*;

public class NeuralNetwork {
    private final int inputCount;
    private final int hiddenCount;
    private final int outputsCount;

    private final double[][] inputToHidden; //Todo: More hidden layers
    private final double[] biasesHidden;
    private final double [][] hiddenToOutput;
    private final double [] biasesOutput;

    private final double [][] outputWeightChanges;
    private final double [][] hiddenWeightsChanges;

    private NeuralNetwork neuralNetwork;
    public NeuralNetwork(int inputCount, int hiddenCount, int outputsCount) {
        this.inputCount = inputCount;
        this.hiddenCount = hiddenCount;
        this.outputsCount = outputsCount;
        this.inputToHidden = new double[inputCount][hiddenCount];
        this.biasesHidden = new double[hiddenCount];
        this.hiddenToOutput = new double[hiddenCount][outputsCount];
        this.biasesOutput = new double[outputsCount];
        this.outputWeightChanges = new double[hiddenCount][outputsCount];
        this.hiddenWeightsChanges = new double[inputCount][hiddenCount];
        initRandom(inputToHidden, hiddenToOutput, biasesHidden, biasesOutput); //init random weights and biases for faster learning
        neuralNetwork = this;
    }

    public double[] run (double[] inputData){
        if(inputData.length < 1 || inputData.length < inputCount) {
            throw new IllegalArgumentException("Input data size too small");
        }
        return forward(forward(inputData, inputToHidden, biasesHidden, hiddenCount), hiddenToOutput, biasesOutput, outputsCount);
    }

    private double[] forward(double[] inputData, double[][] weights, double[] biases, int layerCount){
        double[] layerData = new double[layerCount];

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                double dataPass = inputData[i] * weights[i][j];
                layerData[j] += dataPass;
            }
        }
        for(int i = 0; i < layerCount; i ++){
            layerData[i] += biases[i];
            layerData[i] = UtilMath.sigmoid(layerData[i]); // Activation function
            //Can be replaced with tanh, etc..
        }
        return layerData;
    }

    int iterations = 0;
    public void train(double[][] inputs, double [][] expectedOutputs, double maxErrorRate, double learningRate, double momentumRate){
        double errorRate = 0;
        double errors = 0;
        iterations ++;
        for (int j = 0; j < inputs.length; j++) {
            errors += trainAi(inputs[j], expectedOutputs[j], learningRate, momentumRate);
        }
        errorRate = errors / inputs.length;

        //TODO: Train until max error rate
    }

    public void savetoFile(String path){
        Gson gson = new Gson();
        try {
            Writer writer = new FileWriter(path);
            gson.toJson(neuralNetwork, writer);
            writer.flush();
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    public static NeuralNetwork loadFromFile(String path) throws FileNotFoundException {
        Gson gson = new Gson();
        JsonReader jsonReader = new JsonReader(new FileReader(path));
        return gson.fromJson(jsonReader, NeuralNetwork.class);
    }
    public double trainAi(double[] inputs, double[] expectedValues, double learningRate, double momentumRate) {
        double[] hiddenUnits = forward(inputs, inputToHidden, biasesHidden, hiddenCount);
        double[] predictedValues = forward(hiddenUnits, hiddenToOutput, biasesOutput, outputsCount);

        double[] outputGradients = new double[outputsCount];

        for (int i = 0; i < outputsCount; i++) {
            double outputDelta = expectedValues[i] - predictedValues[i];
            outputGradients[i] = outputDelta * UtilMath.sigmoidDerivative(predictedValues[i]);
        }

        double[] hiddenGradients = new double[hiddenCount];

        for (int i = 0; i < hiddenCount; i++) {
            hiddenGradients[i] = 0;
            for (int j = 0; j < outputsCount; j++) {
                hiddenGradients[i] += outputGradients[j] * hiddenToOutput[i][j];
            }
            hiddenGradients[i] *= UtilMath.sigmoidDerivative(hiddenUnits[i]);
        }
        for (int i = 0; i < hiddenCount; i++) {
            for (int j = 0; j < outputsCount; j++) {
                double weightChange = outputGradients[j] * hiddenUnits[i];
                hiddenToOutput[i][j] += learningRate * weightChange + momentumRate * outputWeightChanges[i][j];
                outputWeightChanges[i][j] = weightChange;
            }
        }
        for (int i = 0; i < inputCount; i++) {
            for (int j = 0; j < hiddenCount; j++) {
                double weightChange = hiddenGradients[j] * inputs[i];
                inputToHidden[i][j] += learningRate * weightChange + momentumRate * hiddenWeightsChanges[i][j];
                hiddenWeightsChanges[i][j] = weightChange;
            }
        }
        return UtilMath.computeMSE(predictedValues, expectedValues);
    }
    private void initRandom(double[][] inputToHidden, double[][] hiddenToOutput, double[] biasesHidden, double[] biasesOutput){
        for (int i = 0; i < inputToHidden.length; i++) {
            for (int j = 0; j < inputToHidden[i].length; j++) {
                inputToHidden[i][j] = UtilMath.generateRandomBias(-0.5, 0.5);
            }
        }
        for (int i = 0; i < hiddenToOutput.length; i++) {
            for (int j = 0; j < hiddenToOutput[i].length; j++) {
                hiddenToOutput[i][j] = UtilMath.generateRandomBias(-0.5, 0.5);
            }
        }
        for(int i = 0; i < biasesHidden.length; i ++){
            biasesHidden[i] = UtilMath.generateRandomBias(-0.5, 0.5);
        }
        for(int i = 0; i < biasesOutput.length; i ++){
            biasesOutput[i] = UtilMath.generateRandomBias(-0.5, 0.5);
        }

    }
}
