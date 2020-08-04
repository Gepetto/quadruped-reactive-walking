#include <cstdlib>
#include <iostream>

#include <Eigen/Core>
#include "example-adder/gepadd.hpp"
#include "example-adder/MPC.hpp"

int main(int argc, char** argv) {
  if (argc == 3) {
    int a = std::atoi(argv[1]), b = std::atoi(argv[2]);
    std::cout << "The sum of " << a << " and " << b << " is: ";
    std::cout << gepetto::example::add(a, b) << std::endl;

    MPC test(0.001f, 2, 0.32f);
    test.create_matrices();
    std::cout << test.gethref() << std::endl;
    /*std::cout << test.getA() << std::endl;
    std::cout << test.getML() << std::endl;
    std::cout << test.getDay() << std::endl;
    std::cout << test.getMonth() << std::endl;
    std::cout << test.getYear() << std::endl;*/

    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    return EXIT_SUCCESS;
  } else {
    std::cerr << "This program needs 2 integers" << std::endl;
    return EXIT_FAILURE;
  }
}
