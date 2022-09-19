#pragma once


namespace vjms {

using ActivationFunctionSignature = float (*)(float);

float fast_sigmoid(float value);
float sigmoid(float value);
float relu(float value);
float linear(float value);
float binary_step(float value);


}// namespace dynann
