

#include "src/dynamic_neural_network.hpp"


#include <iostream>
#include <random>


int main(int, char **)
{
  vjms::DynamicNeuralNetwork dnn;
  dnn.add_inputs(10);
  dnn.add_outputs(10);

  for (int i = 0; i < 1000; i++) {
    dnn.evolve();
    std::cout << "|";
    std::flush(std::cout);
  }
  std::cout << "\n";
 
  // auto &inputs = dnn.get_inputs();
  // auto &outputs = dnn.get_outputs();
}
