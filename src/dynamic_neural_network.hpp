#pragma once

#include <vector>
#include <memory>

#include "node.hpp"

namespace vjms {

class Edge;

class DynamicNeuralNetwork
{
public:
  void add_inputs(size_t count);
  void add_outputs(size_t count);
  std::shared_ptr<Node> add_node(Node::Type type, float value = 1.f, float bias = 0.f, ActivationFunctionSignature function = fast_sigmoid);
  void add_edge(std::shared_ptr<Node> input, std::shared_ptr<Node> output, float weight = 1.f);

  void run();
  void evolve();

  const std::vector<std::shared_ptr<Node>> &get_inputs() const;
  const std::vector<std::shared_ptr<Node>> &get_outputs() const;

private:
  std::shared_ptr<Node> get_random_node();
  std::shared_ptr<Node> get_random_non_input_node();

  std::vector<std::shared_ptr<Node>> m_inputs;
  std::vector<std::shared_ptr<Node>> m_inners;
  std::vector<std::shared_ptr<Node>> m_outputs;

  std::vector<std::shared_ptr<Edge>> m_edges;
};

}// namespace vjms
