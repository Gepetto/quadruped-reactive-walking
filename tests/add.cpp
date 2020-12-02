#include <cassert>
#include "quadruped-reactive-walking/gepadd.hpp"

int main() {
  assert(gepetto::example::add(1, 2) == 3);
  assert(gepetto::example::add(5, -1) == 4);
  assert(gepetto::example::add(-3, -1) == -4);
  return 0;
}
