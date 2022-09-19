#pragma once

#include "node.hpp"
#include "edge.hpp"

#include <list>
#include <memory>


namespace vjms {

class Node;

class DynamicNeuralNetwork
{
public:
  void add_node(Node::Type type, float value = 1.f, float bias = 1.f, ActivationFunctionSignature function = fast_sigmoid);
  void add_edge(std::shared_ptr<Node> input, std::shared_ptr<Node> output, float weight = 1.f);

  void run();

  void evolve();

private:
  std::list<std::shared_ptr<Node>> m_nodes;
  std::list<std::shared_ptr<Edge>> m_edges;
};

}// namespace vjms
