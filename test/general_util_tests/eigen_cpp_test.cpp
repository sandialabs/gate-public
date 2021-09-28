#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <gtest/gtest.h>
#include <array>


TEST(Eigen, TensorSlice) {
	Eigen::Tensor<double, 3> m(3, 10, 10);          //Initialize
	m.setRandom();                               //Set random values 
	std::array<long, 3> offset = { 0,0,0 };         //Starting point
	std::array<long, 3> extent = { 1,10,10 };       //Finish point
	std::array<long, 2> shape2 = { 10,10 };         //Shape of desired rank-2 tensor (matrix)
	std::cout << m.slice(offset, extent).reshape(shape2) << std::endl;  //Extract slice and reshape it into a 10x10 matrix.
}