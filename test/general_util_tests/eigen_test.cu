#include <gtest/gtest.h>
#include <gpu_err_chk.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <array>

#include <type_traits>
#include <typeinfo>

TEST(CudaEigen, Broadcasting) {
	Eigen::Matrix<float, 3, 2> A;
	A << 1, 2, 2, 2, 3, 5;
	Eigen::Matrix<float, 3, 1> B;
	B << 1, 2, 3;
	Eigen::Matrix<float, 3, 2> C = A.array().colwise() * B.array();
	Eigen::Matrix<float, 3, 2> C_known;
	C_known << 1, 2, 4, 4, 9, 15;
	EXPECT_TRUE(C.isApprox(C_known));
}

__global__ void cu_dot(Eigen::Vector3f* v1, Eigen::Vector3f* v2, float* out)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	out[idx] = v1[0].dot(v2[0]);
	return;
}
TEST(CudaEigen, CuDotTest) {
	const int N = 3;
	Eigen::Vector3f vector;
	vector << 1, 2, 3;
	Eigen::Vector3f* dev_vector;
	cudaMalloc((void**)&dev_vector, sizeof(Eigen::Vector3f));
	cudaMemcpyAsync(dev_vector, &vector, sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice, 0);
	CudaCheckError();

	float out[N];
	float* dev_out;
	cudaMalloc((void**)&dev_out, sizeof(float)*N);

	cu_dot<<<N, 1 >>>(dev_vector, dev_vector, dev_out);
	CudaCheckError();

	cudaMemcpyAsync(out, dev_out, sizeof(float)*N, cudaMemcpyDeviceToHost, 0);

	for (int i = 0; i < N; ++i) {
		//std::cout << i << ": " << out[i] << std::endl;
		EXPECT_FLOAT_EQ(out[i], 14.f) << i;
	}
}

__global__ void cu_inverse(const Eigen::Matrix<float, 3, 3>* m, Eigen::Matrix<float, 3, 3>* m_out) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	//Eigen::Matrix<float, 3, 3> m2;
	if (idx < 1) {
		 //m[0].inverse();  // Causes compilation issues due to windows compatibility
	}
}

//TEST(CudaEigen, CuInverseTest) {
//	const int N = 5; // Number of threads
//	Eigen::Matrix<float, 3, 3> m = 3 * Eigen::Matrix<float, 3, 3>::Identity();
//	std::cout << "Initial matrix: " << std::endl << m << std::endl;
//
//	// Allocate device memory
//	FAIL() << "TODO see how we can leverage eigen to solve linear systems per thread on the GPU";
//}

__global__ void cu_matrix_multiply(const Eigen::Matrix<float, 2, 3>* m,
	const Eigen::Matrix<float, 3, 1>* v1,
	const Eigen::Matrix<float, 3, 1>* v2,
	Eigen::Matrix<float, 2, 1>* v_out) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < 1) {
		v_out[0] = (m[0] * (v1[0] - v2[0]));
	}
}

TEST(CudaEigen, CuMatrixMultiplyTest) {
	const int N = 8;
	Eigen::Matrix<float, 2, 3> m1;
	m1 << 1, 2, 3, 3, 4, 5;

	Eigen::Matrix<float, 3, 1> v1, v2;
	v1 << 4, 5, 6;
	v2 << 1, 4, 3;

	Eigen::Matrix<float, 2, 1> v_out;


	// Test the following operation
	// find acceleration error in normal and lateral directions
	//**e_dot2 = M*(a_star-acc_e);
	auto v3_known = m1 * (v1 - v2);

	// Allocate Device Memory
	Eigen::Matrix<float, 2, 3>* m_device;
	Eigen::Matrix<float, 3, 1>* v1_device;
	Eigen::Matrix<float, 3, 1>* v2_device;
	Eigen::Matrix<float, 2, 1>* v_out_device;

	cudaMalloc((void**)&m_device, sizeof(Eigen::Matrix<float, 2, 3>));
	cudaMalloc((void**)&v1_device, sizeof(Eigen::Matrix<float, 3, 1>));
	cudaMalloc((void**)&v2_device, sizeof(Eigen::Matrix<float, 3, 1>));
	cudaMalloc((void**)&v_out_device, sizeof(Eigen::Matrix<float, 2, 1>));

	// Copy Host to device memory
	cudaMemcpy(m_device , &m1, sizeof(Eigen::Matrix<float, 2, 3>), cudaMemcpyHostToDevice);
	cudaMemcpy(v1_device, &v1, sizeof(Eigen::Matrix<float, 3, 1>), cudaMemcpyHostToDevice);
	cudaMemcpy(v2_device, &v2, sizeof(Eigen::Matrix<float, 3, 1>), cudaMemcpyHostToDevice);
	CudaCheckError();

	// Launch the kernel
	cu_matrix_multiply<<<N,1>>>(m_device, v1_device, v2_device, v_out_device);
	CudaCheckError();

	// Copy device memory back to host
	cudaMemcpy(&v_out, v_out_device, sizeof(Eigen::Matrix<float, 2, 1>), cudaMemcpyDeviceToHost);
	CudaCheckError();

	ASSERT_TRUE(v_out.isApprox(v3_known));


	// Free device memory
	cudaFree(m_device);
	cudaFree(v1_device);
	cudaFree(v2_device);
	cudaFree(v_out_device);
}

template<class T>
bool isTensorEqual(const T& t1, const T& t2, const float tol = 1e-6) {
	Eigen::Tensor<float, 0> check = (t1 - t2).sum();
	return (check.data()[0] < tol);
}

TEST(CudaEigen, TensorChip) {

	Eigen::Tensor<float, 3> a(4, 3, 2);
	a.setValues({ {{0,1}, {100,101}, {200,201}}, {{300,301}, {400,401}, {500,501}},
				 {{600, 601}, {700,701}, {800,801}}, {{900,901}, {1000,1001}, {1100,1101}} });
	// THIS CREATES COPIES!!
	Eigen::Tensor<float, 2> a1 = a.chip(2, 0); // a[2,:,:]
	Eigen::Tensor<float, 2> a2 = a.chip(2, 1); // a[:,2,:]
	Eigen::Tensor<float, 2> a3 = a.chip(0, 2); // a[:,:,0]

	// Does this create a copy? -> YES
	// We can save it to an auto to have the chip operations stored, then evaluated only when needed.
	Eigen::Tensor<float, 1> a4 = a.chip(0, 2).chip(1, 1); // a[:,1,0]

	//std::cout << "a" << std::endl << a << std::endl;
	//std::cout << "a[2,:,:]" << std::endl << a1 << std::endl;
	//std::cout << "a[:,2,:]" << std::endl << a2 << std::endl;
	//std::cout << "a[:,:,0]" << std::endl << a3 << std::endl;
	//std::cout << "a[:,1,0]" << std::endl << a4 << std::endl;

	Eigen::Tensor<float, 1> a_expected(4);
	a_expected.setValues({ 100, 400, 700, 1000 });
	//std::cout << "a_expected" << std::endl << a_expected << std::endl;
	ASSERT_TRUE(isTensorEqual(a4, a_expected));

}

__global__ void cu_tensor_chip(const Eigen::TensorFixedSize<float, Eigen::Sizes<4, 3, 2>>* m,
	Eigen::TensorFixedSize<float, Eigen::Sizes<4>>* m_out)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < 1) {
		m_out[0] = m[0].chip(0, 2).chip(1, 1); // m[:,1,0]
	}
}


TEST(CudaEigen, KernelTensorChip) {
	Eigen::TensorFixedSize<float, Eigen::Sizes<4,3,2>> m;
	m.setValues({ {{0,1}, {100,101}, {200,201}}, {{300,301}, {400,401}, {500,501}},
				 {{600, 601}, {700,701}, {800,801}}, {{900,901}, {1000,1001}, {1100,1101}} });
	//std::cout << m << std::endl;
	Eigen::TensorFixedSize<float, Eigen::Sizes<4>> m_out;
	m_out.setZero();
	//std::cout << m_out << std::endl;

	Eigen::TensorFixedSize<float, Eigen::Sizes<4,3,2>>* m_device;
	Eigen::TensorFixedSize<float, Eigen::Sizes<4>>* m_out_device;

	cudaMalloc((void**)&m_device, sizeof(Eigen::TensorFixedSize<float, Eigen::Sizes<4,3,2>>));
	cudaMalloc((void**)&m_out_device, sizeof(Eigen::TensorFixedSize<float, Eigen::Sizes<4>>));
	CudaCheckError();

	cudaMemcpy(m_device, &m, sizeof(Eigen::TensorFixedSize<int, Eigen::Sizes<4, 3, 2>>), cudaMemcpyHostToDevice);
	CudaCheckError();

	cu_tensor_chip<<<1,1>>>(m_device, m_out_device);
	CudaCheckError();

	cudaMemcpy(&m_out, m_out_device, sizeof(Eigen::TensorFixedSize<float, Eigen::Sizes<4>>), cudaMemcpyDeviceToHost);
	CudaCheckError();

	//std::cout << m_out << std::endl;

	cudaFree(m_device);
	cudaFree(m_out_device);

	Eigen::TensorFixedSize<float, Eigen::Sizes<4>> m_out_expected;
	m_out_expected.setValues({ 100, 400, 700, 1000 });
	ASSERT_TRUE(isTensorEqual(m_out, m_out_expected));
}

TEST(CudaEigen, TensorCastToMatrix) {
	Eigen::Tensor<int, 3> a(4, 3, 2);
	a.setValues({ {{0,1}, {100,101}, {200,201}}, {{300,301}, {400,401}, {500,501}},
				 {{600, 601}, {700,701}, {800,801}}, {{900,901}, {1000,1001}, {1100,1101}} });

	//Eigen::Tensor<int, 2> a1 = a.chip(2, 0);

	Eigen::Map<Eigen::Matrix<int, 4, 3>> a1(&a(0, 0, 1));

	Eigen::Matrix<int, 4, 3> a2;
	a2.setOnes();

	//std::cout << a1 + a2 << std::endl;
	//std::cout << &a1.data()[2] << std::endl;
	//std::cout << &a.data()[14] << std::endl;
}

//__global__ void cu_tensor_cast_to_matrix(Eigen::TensorFixedSize<float, Eigen::Sizes<4, 3, 2>* tensor_in,
//	Eigen::Matrix<float, 4, 3>* matrix_math_out, bool no_copies_made = false) {
__global__ void cu_tensor_cast_to_matrix() {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < 1) {
		Eigen::TensorFixedSize<float, Eigen::Sizes<4,3,2>> tensor_in;
		// TODO setting values fails in CUDA
		//tensor_in.setValues({ {{0,1}, {100,101}, {200,201}}, {{300,301}, {400,401}, {500,501}},
		//			 {{600, 601}, {700,701}, {800,801}}, {{900,901}, {1000,1001}, {1100,1101}} });
		//Eigen::Matrix<float, 4, 3> x;
		//x.setOnes();
		//Eigen::Map<Eigen::Matrix<float, 4, 3>> y(&tensor_in.data()[14]);

		// Check if there was a copy made
		// printf("Tensor input location: %p: \n", &tensor_in.data()[14]);
		// printf("Matrix mapped location: %p: \n", &y.data()[2]);

		//if (&tensor_in.data()[14] == &y.data()[2]) {
		//	//no_copies_made = true;
		//	printf("No copies made!\n");
		//}

		//*matrix_math_out = x + y;
	}
}

//TEST(CudaEigen, KernelTensorToMatrix) {
//	//Eigen::Tensor<float, 3> a(4, 3, 2);
//	//a.setValues({ {{0,1}, {100,101}, {200,201}}, {{300,301}, {400,401}, {500,501}},
//	//			 {{600, 601}, {700,701}, {800,801}}, {{900,901}, {1000,1001}, {1100,1101}} });
//	cu_tensor_cast_to_matrix<<<1,1>>>();
//	CudaCheckError();
//	FAIL();
//}