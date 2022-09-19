#include "activation_function.hpp"

#include <cmath>


namespace vjms {

float fast_sigmoid(float value)
{
  return value / (1.f + std::abs(value));
}

float sigmoid(float value)
{
  return 1.f / (1.f + std::exp(value));
}

float relu(float value)
{
  return std::max(0.f, value);
}

float linear(float value)
{
  return value;
}

float binary_step(float value)
{
  return (value <= 0.f) ? 0.f : 1.f;
}

}// namespace dynann