package me.vjta088x.ai;

import me.vjta088x.ai.network.NeuralNetwork;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws IOException {
        NeuralNetwork neuralNetwork = null;
        try{
            neuralNetwork = NeuralNetwork.loadFromFile("networkSave.json");
        }catch (FileNotFoundException e){
            neuralNetwork = new NeuralNetwork(2 ,3,1);
        }
        if(neuralNetwork == null){
            neuralNetwork = new NeuralNetwork(2 ,3,1);
        }
        //Loading of network from file

        double[][] inputs = {{0,1},{0,0}, {1,1}, {1,0}}; //simple xor
        double[][] targets = {{1}, {0}, {0}, {1}};

        for (int i = 0; i < 100000; i++){
            neuralNetwork.train(inputs, targets, 0.01, 0.01, 0.9);
        }

        System.out.println("Expected outputs: ");
        for (double[] target : targets) {
            System.out.println(Arrays.toString(target));
        }
        System.out.println("Predicted outputs");
        for (double[] input : inputs) {
            System.out.println(Arrays.toString(neuralNetwork.run(input)));
        }
        neuralNetwork.savetoFile("networkSave.json"); //Serialization of network to file
    }
}