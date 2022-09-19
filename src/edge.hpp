#pragma once


#include <memory>

namespace vjms {

class Node;

class Edge
{
public:
  void set_input(std::shared_ptr<Node> input);
  std::shared_ptr<Node> get_input() const;
  void set_output(std::shared_ptr<Node> output);
  std::shared_ptr<Node> get_output() const;

  float get_weight() const;
  void set_weight(float weight);

private:
  float m_weight = 1.f;

  std::shared_ptr<Node> m_input;
  std::shared_ptr<Node> m_output;
};

}// namespace vjms
