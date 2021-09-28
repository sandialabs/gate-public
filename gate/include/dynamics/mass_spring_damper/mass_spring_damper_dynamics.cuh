#ifndef MSD_DYNAMICS_CUH_
#define MSD_DYNAMICS_CUH_

#include <dynamics/dynamics_stream_managed.cuh>
#include <perturbations/perturbations_stream_managed.cuh>
#include <perturbations/mass_spring_damper/mass_spring_damper_perturbations.cuh>

namespace msd {
	const int S_DIM = 2;
	const int C_DIM = 1;
}

struct MsdParams {
	float m = 1; //mass
	float k = 10; // spring
	float b = 1; // damper

	// dynamic arrays
	Eigen::Matrix<float, 2, 2> A;
	Eigen::Matrix<float, 2, 1> B;

	std::array<float2, 1> ctrl_limits;	
	MsdParams() {
		A << 0, 1,
			(-k / m), (-b / m);
		B << 0,
			1 / m;

	};
	~MsdParams() = default;
};

class MsdDynamics : public GATE_internal::Dynamics<MsdDynamics, MsdParams, msd::S_DIM, msd::C_DIM>
{
public:
	MsdDynamics(cudaStream_t stream = nullptr) : 
		GATE_internal::Dynamics<MsdDynamics, MsdParams, msd::S_DIM, msd::C_DIM>(stream)
	{
		this->params_ = MsdParams();
	}
	MsdDynamics(MsdParams& params, cudaStream_t stream = nullptr) :
		GATE_internal::Dynamics<MsdDynamics, MsdParams, msd::S_DIM, msd::C_DIM>(params.ctrl_limits, stream) {
		this->params_ = params;
	}
	~MsdDynamics() = default;

	// x(0) = x
	// x(1) = x_dot
	__host__ __device__ void computeStateDeriv(state_array& x, control_array& u, state_array& xdot) {

		xdot = this->params_.A * x + this->params_.B * u(0);

	}
	template<class PER_T>
	__host__ __device__ void computeStateDeriv(PER_T* pert, int timestep, int rollout, state_array& x, control_array& u, state_array& xdot) {

		float x1 = x(0);
		float x2 = x(1);
		float k = this->params_.k;
		float m = this->params_.m;
		float b = this->params_.b;

		float check = m;

		// do not perturb first rollout
		if (rollout > 0) {
			// mass perturbation
			m = m  + (*(pert->mass_pert_d_))(rollout); 

		}

		xdot(0) = x2;
		xdot(1) = -(k/m) * x1 - (b/m) * x2 + (1/m) * u(0);




	}
};
#endif // MSD_DYNAMICS_CUH_