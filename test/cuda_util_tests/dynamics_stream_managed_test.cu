#include <gtest/gtest.h>
#include <cuda_util/stream_managed.cuh>
#include <dynamics/dynamics_stream_managed.cuh>

namespace mock {
	const int S_DIM = 4;
	const int C_DIM = 2;
}
// Create a mock parameter object
struct MockParams {
	int x = 5;
	int y = 10;
	MockParams() = default;
	~MockParams() = default;
};

// Create a mock object that inherits from dynamics
class MockDynamics : public GATE_internal::Dynamics<MockDynamics, MockParams, mock::S_DIM, mock::C_DIM> {
public:
	MockDynamics(cudaStream_t stream = nullptr) :
		GATE_internal::Dynamics<MockDynamics, MockParams, mock::S_DIM, mock::C_DIM>(stream) {
		this->params_ = MockParams();
	}

	MockDynamics(std::array<float2, mock::C_DIM> ctrl_ranges, cudaStream_t stream = nullptr) :
		GATE_internal::Dynamics<MockDynamics, MockParams, mock::S_DIM, mock::C_DIM>(ctrl_ranges, stream) {
		this->params_ = MockParams();
	}

	~MockDynamics() = default;
};


TEST(DynamicsStreamManaged, Construction) {
	auto A = MockDynamics();
}

TEST(DynamicsStreamManaged, GetControlRanges_Max) {
	auto A = MockDynamics();
	auto ctrl_range = A.getControlRanges();
	for (int i = 0; i < mock::C_DIM; ++i) {
		ASSERT_EQ(-FLT_MAX, ctrl_range[i].x);
		ASSERT_EQ(FLT_MAX, ctrl_range[i].y);
	}
}

TEST(DynamicsStreamManaged, GetControlRanges_Fixed) {
	std::array<float2, 2> input_ranges;
	input_ranges[0].x = -3;
	input_ranges[0].y = 2;
	input_ranges[1].x = 1;
	input_ranges[1].y = 21;
	auto A = MockDynamics(input_ranges);
	auto ctrl_range = A.getControlRanges();
	for (int i = 0; i < mock::C_DIM; ++i) {
		ASSERT_EQ(input_ranges[i].x, ctrl_range[i].x);
		ASSERT_EQ(input_ranges[i].y, ctrl_range[i].y);
	}
}

TEST(DynamicsStreamManaged, GPUSetupFlags) {
	auto* A = new MockDynamics();
	ASSERT_FALSE(A->getGPUMemStatus());
	A->GPUSetup();
	ASSERT_TRUE(A->getGPUMemStatus());
}

TEST(DynamicsStreamManaged, CPUParams) {
	auto A = MockDynamics();
	auto default_params = A.getParams();
	ASSERT_EQ(default_params.x, 5);
}

TEST(DynamicsStreamManaged, SetParams) {
	auto A = MockDynamics();
	auto new_params = MockParams();
	new_params.x = 7;
	A.setParams(new_params);
	auto set_params = A.getParams();
	ASSERT_EQ(new_params.x, set_params.x);
}