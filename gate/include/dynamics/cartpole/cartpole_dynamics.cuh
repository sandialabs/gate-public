#ifndef CARTPOLE_DYNAMICS_CUH_
#define CARTPOLE_DYNAMICS_CUH_

#include <dynamics/dynamics_stream_managed.cuh>
#include <perturbations/perturbations_stream_managed.cuh>
#include <perturbations/cartpole/cartpole_perturbations.cuh>


namespace cartpole {
	const int S_DIM = 4;  // state dimension
	const int C_DIM = 1;  // control dimension
}

struct CartpoleParams {
	float mc = 1.f;  // mass cart
	float mp = .5f;  // mass pole
	float l = 1.f;  // length to c.o.m. pole
	std::array<float2, 1> ctrl_limits;
	CartpoleParams() {};
	~CartpoleParams() = default;
};


class CartpoleDynamics : public GATE_internal::Dynamics<CartpoleDynamics, CartpoleParams, cartpole::S_DIM, cartpole::C_DIM> {
public:
	CartpoleDynamics(cudaStream_t stream = nullptr) : 
		GATE_internal::Dynamics<CartpoleDynamics, CartpoleParams, cartpole::S_DIM, cartpole::C_DIM>(stream) {
			this->params_ = CartpoleParams();
		}
	CartpoleDynamics(CartpoleParams& params, cudaStream_t stream = nullptr) :
		GATE_internal::Dynamics<CartpoleDynamics, CartpoleParams, cartpole::S_DIM, cartpole::C_DIM>(params.ctrl_limits, stream) {
			this->params_ = params;
		}
	~CartpoleDynamics() = default;

	// x(0) = x
	// x(1) = x_dot
	template<class PER_T>
	__host__ __device__ void computeStateDeriv(PER_T* pert, int timestep, int rollout, state_array& x, control_array& u, state_array& xdot) {

		float mc = this->params_.mc;  // mass cart
		float mp = this->params_.mp;  // mass pole
		float l = this->params_.l;  // length to c.o.m. pole
		float g = 9.8f;  // gravity
		float M = mc + mp;

		// do not perturb first rollout
		if (rollout > 0) {
			// mass perturbation
			mc = abs(mc + (*(pert->mc_pert_d_))(rollout));
			mp = abs(mp + (*(pert->mp_pert_d_))(rollout));
			l  = abs(l  + (*(pert->l_pert_d_ ))(rollout));
			M = mc + mp;
		}

		//x, xd, th, thd = state
		xdot(0) = x(1);
		xdot(1) = (mp * g * cos(x(2)) * sin(x(2)) - mp * l * x(3) * x(3) * sin(x(2)) + u(0)) / (M - mp * cos(x(2)) * cos(x(2)));
		xdot(2) = x(3);
		xdot(3) = (M * g * sin(x(2)) - mp * l * x(3) * x(3) * sin(x(2)) * cos(x(2)) + u(0) * cos(x(2))) / (M * l - mp * l * cos(x(2)) * cos(x(2)));
	
	}
};
#endif // CARTPOLE_DYNAMICS_CUH_