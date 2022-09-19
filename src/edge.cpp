#include "edge.hpp"

namespace vjms {

void Edge::set_input(std::shared_ptr<Node> input)
{
  m_input = input;
}
std::shared_ptr<Node> Edge::get_input() const
{
  return m_input;
}
void Edge::set_output(std::shared_ptr<Node> output)
{
  m_output = output;
}
std::shared_ptr<Node> Edge::get_output() const
{
  return m_output;
}
float Edge::get_weight() const
{
  return m_weight;
}
void Edge::set_weight(float weight)
{
  m_weight = weight;
}

}// namespace vjms