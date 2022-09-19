#include "dynamic_neural_network.hpp"


namespace vjms {

void DynamicNeuralNetwork::add_node(Node::Type type, float value, float bias, ActivationFunctionSignature function)
{
  auto node = std::make_shared<Node>();
  node->set_type(type);
  node->set_value(value);
  node->set_bias(bias);
  node->set_activation_function(function);
  m_nodes.emplace_back(node);
}

void DynamicNeuralNetwork::add_edge(std::shared_ptr<Node> input, std::shared_ptr<Node> output, float weight)
{
  auto edge = std::make_shared<Edge>();
  edge->set_input(input);
  edge->set_output(output);
  edge->set_weight(weight);
  m_edges.emplace_back(edge);
}

// O(n)
// This has a lag built-in, since the value calculations for the nodes are done only after every edge has been visited.
// But as this is used for active time applications, where the inputs are constantly changing, it should be fine.
// The human brain, too, has lag for every decision it makes. The deeper the network is, the greater the lag will be.
// The reason for doing it this way is to enable looping connections
void DynamicNeuralNetwork::run()
{
  for (auto node : m_nodes) {
    node->get_accumulator().reset();
  }

  for (auto edge : m_edges) {
    auto input = edge->get_input();
    auto output = edge->get_output();
    output->get_accumulator().add(edge->get_weight() * input->get_value());
  }

  for (auto node : m_nodes) {
    // Ignore input types, because their values are just set as is.
    if (node->get_type() != Node::Type::Input) {
      auto accumulation = node->get_accumulator().get_value();
      auto activation = node->get_activation_function();
      node->set_value(activation(accumulation + node->get_bias()));
    }
  }
}


}// namespace vjms