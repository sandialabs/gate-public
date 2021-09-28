#include <gtest/gtest.h>
#include <cuda_util/stream_managed.cuh>
#include <controllers/guidance_stream_managed.cuh>
#include <dynamics/dynamics_stream_managed.cuh>
#include <dynamics/integrators_eigen.cuh>
#include <perturbations/perturbations_stream_managed.cuh>
#include <cuda_util/cuda_memory_utils.cuh>

namespace mock {
	const int S_DIM = 4;
	const int C_DIM = 2;
	const int N_TIMESTEPS = 100;
	const int N_ROLLOUTS = 10;
}
// Create a mock dynamics parameter object
struct MockParams {
	int mass = 5;
	int length = 10;
	float gas_constant = 33;
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

	void computeStateDeriv(MockDynamics::state_array& x, MockDynamics::control_array& u, MockDynamics::state_array& xdot) {

		xdot(0) = 1.f;
		xdot(1) = params_.mass*x(2) + u(0);
		xdot(2) = params_.length*x(3) + u(1);
		xdot(3) = x(3); 

	}

	// TODO UNTESTED
	template<class PERT_T>
	__device__ void computeStateDeriv(PERT_T* perturbations, int timestep, int rollout, MockDynamics::state_array& x, MockDynamics::control_array& u, MockDynamics::state_array& xdot) {
		xdot(0) = 1.f;
		xdot(1) = (params_.mass + perturbations->mass_pert_d_)*x(2) + u(0) + perturbations->u_t_pert_d_(0, rollout + timestep*mock::N_ROLLOUTS);
		xdot(2) = (params_.length + perturbations->length_pert_d_)*x(3) + u(1) + perturbations->u_t_pert_d_(1, rollout + timestep * mock::N_ROLLOUTS);
		xdot(3) = x(3);
	}

};

class MockPertParams : public GATE_internal::PerturbationParam<mock::S_DIM, mock::C_DIM> {
public:
	MockPertParams() : GATE_internal::PerturbationParam<mock::S_DIM, mock::C_DIM>() {}
	MockPertParams(const state_array& x0_mean, const state_array& x0_std, const control_array& u_std) :
		GATE_internal::PerturbationParam<mock::S_DIM, mock::C_DIM>(x0_mean, x0_std, u_std) {}
	~MockPertParams() = default;

	Eigen::Matrix<float, 3, 1> force_std = Eigen::Vector3f::Zero();
	float mass_std_ = 0;
	float length_std_ = 0;
};

class MockPert : public GATE_internal::Perturbations<MockPert,
	MockPertParams, mock::S_DIM, mock::C_DIM, mock::N_TIMESTEPS, mock::N_ROLLOUTS> {
public:
	MockPert(cudaStream_t stream = nullptr) :
		GATE_internal::Perturbations<MockPert,
		MockPertParams, mock::S_DIM, mock::C_DIM, mock::N_TIMESTEPS, mock::N_ROLLOUTS>(stream) {
		MockPert::allocateCUDAMem();
		//std::cout << "Constructed mockPert" << std::endl;

	}
	MockPert(const MockPertParams& params, cudaStream_t stream = nullptr) :
		GATE_internal::Perturbations<MockPert,
		MockPertParams, mock::S_DIM, mock::C_DIM, mock::N_TIMESTEPS, mock::N_ROLLOUTS>(params, stream) {
		MockPert::allocateCUDAMem();
		//std::cout << "Constructed mockPert" << std::endl;

	}
	~MockPert() {
		MockPert::deallocateCudaMem();
		//std::cout << "Destructed mockPert" << std::endl;

	};

	// Device pointers for MockPert derived perturbations
	scalar_all_rollouts* mass_pert_d_ = nullptr;
	scalar_all_rollouts* length_pert_d_ = nullptr;

	void initPerturbations() {
		scalar_all_rollouts mass_pert_host = scalar_all_rollouts::NullaryExpr(1, mock::N_ROLLOUTS, [&]() { return normal_distribution_(generator_); }); // STATE_DIM x NUM_ROLLOUTS

		// Scale the columns by the correct standard deviations and add the mean
		mass_pert_host = (mass_pert_host * params_.mass_std_); // TODO turn into a util?

		// Transfer the data to the GPU
		HANDLE_ERROR(cudaMemcpyAsync(mass_pert_d_, &mass_pert_host, sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, stream_));
		CudaCheckError();

		scalar_all_rollouts length_pert_host = scalar_all_rollouts::NullaryExpr(1, mock::N_ROLLOUTS, [&]() { return normal_distribution_(generator_); }); // STATE_DIM x NUM_ROLLOUTS

	// Scale the columns by the correct standard deviations and add the mean
		length_pert_host = (length_pert_host * params_.length_std_); // TODO turn into a util?

		// Transfer the data to the GPU
		HANDLE_ERROR(cudaMemcpyAsync(length_pert_d_, &length_pert_host, sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, stream_));
		CudaCheckError();

		// TODO init force distributions.
	}

private:
	void allocateCUDAMem() {
		GATE_internal::cudaObjectMalloc(mass_pert_d_);
		GATE_internal::cudaObjectMalloc(length_pert_d_);
		//std::cout << "Allocated mockPert CUDA memory" << std::endl;
	}

	void deallocateCudaMem() {
		GATE_internal::cudaObjectFree(mass_pert_d_);
		GATE_internal::cudaObjectFree(length_pert_d_);
		//std::cout << "Deallocated mockPert CUDA memory" << std::endl;

	}
};


TEST(General_Util, RK4) {
	//auto* D = new MockDynamics();
	auto D = MockDynamics();

	MockDynamics::state_array x_k;
	MockDynamics::control_array u_k;
	MockDynamics::control_array u_kp1;
	MockDynamics::state_array xdot;
	MockDynamics::state_array x_plus_known;
	MockDynamics::state_array x_plus_compute;

	x_k << 1.5f, 0.3f, 2.1f, -0.9;
	u_k << 0.5f, 1.2f;
	u_kp1 << 4.5f, -1.f; 

	x_plus_known << 2.0000f, 0.41197917f, -3.6859375f, -1.48359375f;

	float dt = 0.5f; 

	integrators_eigen::rk4<MockDynamics>(&D, x_k, u_k, dt, u_kp1, xdot, x_plus_compute);

	for(int i = 0; i < MockDynamics::STATE_DIM; ++i){
		EXPECT_NEAR(x_plus_compute(i), x_plus_known(i), 1e-4);
	}

}

