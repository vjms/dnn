#pragma once


namespace vjms {

class Accumulator
{
public:
  void reset() { m_value = 0.f; }
  void add(float value) { m_value += value; }
  float get_value() const { return m_value; }

private:
  float m_value = 0.f;
};

}// namespace vjms
