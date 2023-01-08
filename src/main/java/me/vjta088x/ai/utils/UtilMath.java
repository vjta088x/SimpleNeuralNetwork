package me.vjta088x.ai.utils;

import java.util.Random;

public class UtilMath {
    static Random random;
    static{
        random = new Random();
    }
    public static double sigmoid(double input){
        return 1/(1+Math.exp(-input));
    }
    public static double sigmoidDerivative(double x) {
        return x * (1 - x);
    }
    public static double generateRandomBias(double minBias, double maxBias){
        return minBias + (maxBias - minBias) * random.nextDouble();
    }

    public static double calculateChange (double learningRate, double delta, double value, double momentum, double pastChange){
        return (learningRate * delta * value) + (momentum * pastChange);
    }

    public static double computeMSE(double[] predicted, double[] trueOutput) {
        double loss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double error = predicted[i] - trueOutput[i];
            loss += error * error;
        }
        return loss / predicted.length;
    }

}
