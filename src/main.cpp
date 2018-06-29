#include <cstdlib>
#include <iostream>

#include "example-adder/gepadd.hpp"

int main(int argc, char** argv) {
  if (argc == 3) {
    int a = std::atoi(argv[1]), b = std::atoi(argv[2]);
    std::cout << "The sum of " << a << " and " << b << " is: ";
    std::cout << gepetto::example::add(a, b) << std::endl;
    return EXIT_SUCCESS;
  } else {
    std::cerr << "This program needs 2 integers" << std::endl;
    return EXIT_FAILURE;
  }
}
