#include "node.hpp"

namespace vjms {

void Node::set_type(Type type)
{
  m_type = type;
}
Node::Type Node::get_type() const
{
  return m_type;
}
void Node::set_value(float value)
{
  m_value = value;
}
float Node::get_value() const
{
  return m_value;
}
void Node::set_bias(float bias)
{
  m_bias = bias;
}
float Node::get_bias() const
{
  return m_bias;
}
ActivationFunctionSignature Node::get_activation_function() const
{
  return m_activation_function;
}
void Node::set_activation_function(ActivationFunctionSignature function)
{
  m_activation_function = function;
}
Accumulator &Node::get_accumulator()
{
  return m_accumulator;
}
void Node::calculate_value()
{
  m_value = m_activation_function(m_accumulator.get_value() + m_bias);
}
}// namespace vjms
