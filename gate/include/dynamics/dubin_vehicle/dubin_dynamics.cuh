#ifndef DUBIN_DYNAMICS_CUH_
#define DUBIN_DYNAMICS_CUH_

#include <dynamics/dynamics_stream_managed.cuh>
#include <perturbations/dubin_vehicle/dubin_perturbations.cuh>

namespace dubin {
	const int S_DIM = 3;
	const int C_DIM = 1;
}


struct DubinParams {
	float V_fixed = 2;
	std::array<float2, 1> ctrl_limits;	
	DubinParams() {

	};
	~DubinParams() = default;
};

class DubinDynamics : public GATE_internal::Dynamics<DubinDynamics, DubinParams, dubin::S_DIM, dubin::C_DIM>
{
public:
	DubinDynamics(cudaStream_t stream = nullptr) : 
		GATE_internal::Dynamics<DubinDynamics, DubinParams, dubin::S_DIM, dubin::C_DIM>(stream)
	{
		this->params_ = DubinParams();
	}
	DubinDynamics(DubinParams& params, cudaStream_t stream = nullptr) :
		GATE_internal::Dynamics<DubinDynamics, DubinParams, dubin::S_DIM, dubin::C_DIM>(params.ctrl_limits, stream) {
		this->params_ = params;
	}
	~DubinDynamics() = default;

	// x(0) = x
	// x(1) = y
	// x(2) = theta
	template<class PER_T>
	__host__ __device__ void computeStateDeriv(PER_T* pert, int timestep, int rollout, state_array& x, control_array& u, state_array& xdot) {
		float V = this->params_.V_fixed;

		xdot(0) = V * cos(x(2));
		xdot(1) = V * sin(x(2));
		xdot(2) = u(0);

	}
};
#endif // DUBIN_DYNAMICS_CUH_