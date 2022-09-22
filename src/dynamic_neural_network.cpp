#include "dynamic_neural_network.hpp"

#include "node.hpp"
#include "edge.hpp"

#include <random>
#include <iostream>
#include <algorithm>

namespace vjms {

static std::random_device rng_dev;
static std::mt19937 rng_gen{ rng_dev() };


void DynamicNeuralNetwork::add_inputs(size_t count)
{
  while (count--) {
    add_node(Node::Type::Input);
  }
}
void DynamicNeuralNetwork::add_outputs(size_t count)
{
  while (count--) {
    add_node(Node::Type::Output);
  }
}
std::shared_ptr<Node> DynamicNeuralNetwork::add_node(Node::Type type, float value, float bias, ActivationFunctionSignature function)
{
  auto node = std::make_shared<Node>();
  node->set_type(type);
  node->set_value(value);
  node->set_bias(bias);
  node->set_activation_function(function);
  switch (type) {
  case Node::Type::Input: m_inputs.emplace_back(node); break;
  case Node::Type::Inner: m_inners.emplace_back(node); break;
  case Node::Type::Output: m_outputs.emplace_back(node); break;
  }
  return node;
}

void DynamicNeuralNetwork::add_edge(std::shared_ptr<Node> input, std::shared_ptr<Node> output, float weight)
{
  if (input == output) {
    return;
  }
  // Do not add the edge if it already exists
  auto it = std::find_if(m_edges.begin(), m_edges.end(), [&input, &output](const auto &edge) {
    // Check both directions
    return (edge->get_input() == input && edge->get_output() == output)
           || (edge->get_input() == output && edge->get_output() == input);
  });
  if (it != m_edges.end()) {
    return;
  }
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
  for (auto node : m_inners) {
    node->get_accumulator().reset();
  }
  for (auto node : m_outputs) {
    node->get_accumulator().reset();
  }

  for (auto edge : m_edges) {
    auto input = edge->get_input();
    auto output = edge->get_output();
    output->get_accumulator().add(edge->get_weight() * input->get_value());
  }

  for (auto node : m_inners) {
    node->calculate_value();
  }
  for (auto node : m_outputs) {
    node->calculate_value();
  }
}

std::shared_ptr<Node> DynamicNeuralNetwork::get_random_node()
{
  std::uniform_int_distribution selector{ (size_t)0, m_inputs.size() + m_inners.size() + m_outputs.size() - (size_t)1 };
  auto index = selector(rng_gen);
  if (index < m_inputs.size()) {
    return m_inputs[index];
  }
  if (index < m_inputs.size() + m_inners.size()) {
    return m_inners[index - m_inputs.size()];
  }
  return m_outputs[index - m_inputs.size() - m_inners.size()];
}
std::shared_ptr<Node> DynamicNeuralNetwork::get_random_non_input_node()
{
  std::uniform_int_distribution selector{ (size_t)0, m_inners.size() + m_outputs.size() - (size_t)1 };
  auto index = selector(rng_gen);
  if (index < m_inners.size()) {
    return m_inners[index];
  }
  return m_outputs[index - m_inners.size()];
}

void DynamicNeuralNetwork::evolve()
{
  // std::uniform_real_distribution dist_real{ -1.f, 1.f };
  std::normal_distribution dist_real{};

  std::bernoulli_distribution new_edge{ 0.5 };
  if (new_edge(rng_gen)) {
    std::shared_ptr<Node> input, output;
    input = get_random_node();
    output = get_random_non_input_node();
    add_edge(input, output, dist_real(rng_gen));
  }


  // Only inner nodes can be created.
  std::bernoulli_distribution new_node{ 0.2 };
  if (new_node(rng_gen)) {
    add_node(Node::Type::Inner, dist_real(rng_gen), dist_real(rng_gen));
  }

  auto random_remove = [](auto &container, float chance) {
    if (container.size() != 0) {
      std::bernoulli_distribution remove{ chance };
      if (remove(rng_gen)) {
        std::uniform_int_distribution index{ (size_t)0, container.size() - 1 };
        container.erase(container.begin() + index(rng_gen));
      }
    }
  };

  random_remove(m_edges, 0.1);
  // Only inner nodes can be removed as inputs and outputs are per definition always in use.
  random_remove(m_inners, 0.1);

  auto adjust_bias = [&](auto &nodes) {
    for (auto &node : nodes) {
      node->set_bias(node->get_bias() + dist_real(rng_gen));
    }
  };
  adjust_bias(m_inputs);
  adjust_bias(m_outputs);
  adjust_bias(m_inners);

  for (auto &edge : m_edges) {
    edge->set_weight(edge->get_weight() + dist_real(rng_gen));
  }
}
const std::vector<std::shared_ptr<Node>> &DynamicNeuralNetwork::get_inputs() const
{
  return m_inputs;
}
const std::vector<std::shared_ptr<Node>> &DynamicNeuralNetwork::get_outputs() const
{
  return m_outputs;
}

}// namespace vjms