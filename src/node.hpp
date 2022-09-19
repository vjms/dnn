#pragma once

#include "activation_function.hpp"
#include "accumulator.hpp"

namespace vjms {

class Node
{
public:
  enum class Type {
    Input,
    Inner,
    Output
  };

  void set_type(Type type);
  Type get_type() const;

  void set_value(float value);
  float get_value() const;

  void set_bias(float bias);
  float get_bias() const;

  ActivationFunctionSignature get_activation_function() const;
  void set_activation_function(ActivationFunctionSignature function);


  Accumulator &get_accumulator();

private:
  ActivationFunctionSignature m_activation_function = fast_sigmoid;
  float m_value = 0.f;
  float m_bias = 0.f;
  Accumulator m_accumulator;
  Type m_type = Type::Input;
};

}// namespace vjms