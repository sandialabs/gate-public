#include <gtest/gtest.h>
#include <trajectory_generation/GATE.cuh>
#include <dynamics/quadrotor/quad_dynamics.cuh>
#include <controllers/quadrotor/quad_LQR.cuh>
#include <memory>
#include <chrono>
#include <iostream>


//template <class T>
//__global__ void testx0Kernel(T::all_states_at_t* x0_noise_device) {
//	int idx = blockDim.x * blockIdx.x + threadIdx.x;
//	printf("Idx: %i, Mean: %f\n", idx, (*x0_noise_device).row(idx).mean());
//}

TEST(GATE_Quad, PropagateTrajectories) {
	// simulation timestep
	float dt = 0.01;

	// Setup rollouts
	const int num_timesteps = 1500;
	const int num_rollouts = 1024;
	const int bdim_x = 64;

	// set params
	std::shared_ptr<QuadDynamics> dyn = std::make_shared<QuadDynamics>();

	// LQR Controller
	std::shared_ptr<QuadLQR> ctrl = std::make_shared<QuadLQR>();
	auto default_guid_params = ctrl->getGuidanceParams();
	//default_guid_params.dt = dt; //todo is this necessary?
	//default_guid_params.setNumTimesteps(num_timesteps);


	auto default_dyn_params = ctrl->getDynamicsParams();
	ctrl->setGuidanceParams(default_guid_params);
	ctrl->setDynamicsParams(default_dyn_params);
	dyn->setParams(default_dyn_params);


	// perturbations
	QuadPertParams::state_array x0_mean, x0_std;
	QuadPertParams::control_array u_std;
	x0_mean << -2.f, -2.f, -2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f;
	x0_std << 0.1f, 0.1f, 0.1f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f;
	//u_std << .0f, .0f, .0f, .0f;
	u_std << 0, 0, .25f, .25f, .25f, .25f; // First two inputs are force in x and y
	QuadPertParams params = QuadPertParams(x0_mean, x0_std, u_std);
	std::shared_ptr<QuadPert<num_timesteps, num_rollouts>> pert = std::make_shared<QuadPert<num_timesteps, num_rollouts>>(params);
	pert->initializeX0andControlPerturbations();
	pert->initPerturbations();


	// Create the GATE object
	typedef GATE<QuadDynamics, QuadLQR, QuadPert<num_timesteps, num_rollouts>, num_timesteps, num_rollouts, bdim_x> Quad_ROTE;
	std::shared_ptr<Quad_ROTE> RTE = std::make_shared<Quad_ROTE>(dyn.get(), ctrl.get(), pert.get(), x0_mean, dt);

	RTE->computeTrajectories();

	// Assert that there are no NaN's
	for (int i = 0; i < num_rollouts; ++i) {
		for (int j = 0; j < QuadDynamics::STATE_DIM; ++j) {
			ASSERT_FALSE(isnan(
				RTE->state_trajectories_host[num_timesteps * QuadDynamics::STATE_DIM * i // Rollout
				+ (num_timesteps - 1) * QuadDynamics::STATE_DIM //Timestep
				+ j])) << "Rollout: " << i << ", State: " << j << std::endl; // State
		}
	}

	
	
}