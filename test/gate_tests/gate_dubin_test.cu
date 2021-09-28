#include <gtest/gtest.h>
#include <trajectory_generation/GATE.cuh>
#include <dynamics/dubin_vehicle/dubin_dynamics.cuh>
#include <controllers/dubin_vehicle/dubin_PID.cuh>
#include <perturbations/dubin_vehicle/dubin_perturbations.cuh>
#include <memory>
#include <chrono>
#include <iostream>


template <class T>
__global__ void testx0Kernel(T::all_states_at_t* x0_noise_device) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	printf("Idx: %i, Mean: %f\n", idx, (*x0_noise_device).row(idx).mean());
}

TEST(GATE_Dubin, PropagateTrajectories) {
	// simulation timestep
	float dt = 0.1;

	// Setup rollouts
	const int num_timesteps = 500; // 2500;
	const int num_rollouts = 8192; // 500;
	const int bdim_x = 64;

	std::shared_ptr<DubinDynamics> dyn = std::make_shared<DubinDynamics>();
	std::shared_ptr<DubinPID> ctrl = std::make_shared<DubinPID>();
	// set params
	auto default_guid_params = ctrl->getGuidanceParams();
	auto default_dyn_params = ctrl->getDynamicsParams();

	default_guid_params.dt = dt;
	default_guid_params.p_gain = 0.85;
	default_guid_params.i_gain = 0.f;
	default_guid_params.pos_desired << 10.f, 10.f;

	ctrl->setGuidanceParams(default_guid_params);
	ctrl->setDynamicsParams(default_dyn_params);
	dyn->setParams(default_dyn_params);

	// perturbations
	DubinPertParams::state_array x0_mean, x0_std;
	DubinPertParams::control_array u_std;
	x0_mean << -5.f, 3.f, 0.f;
	x0_std << 1.f, 1.f, 1.f; // 0, 0, 0;
	u_std << .1;
	DubinPertParams params = DubinPertParams(x0_mean, x0_std, u_std);
	std::shared_ptr<DubinPert<num_timesteps, num_rollouts>> pert = std::make_shared<DubinPert<num_timesteps, num_rollouts>>(params);
	pert->initializeX0andControlPerturbations();
	pert->initPerturbations();

	// Create the GATE object
	typedef GATE<DubinDynamics, DubinPID, DubinPert<num_timesteps, num_rollouts>, num_timesteps, num_rollouts, bdim_x> Dubin_ROTE;
	std::shared_ptr<Dubin_ROTE> RTE = std::make_shared<Dubin_ROTE>(dyn.get(), ctrl.get(), pert.get(), x0_mean, dt);

	RTE->computeTrajectories();
	
	//std::cout << "Trajectory 0: " << std::endl;
	//int idx = 0;
	//float tmp;
	//int sim_idx = 0;
	//for (int i = 0; i < num_timesteps; ++i) {
	//	std::cout << sim_idx << "    ";
	//	for (int j = 0; j < DubinDynamics::STATE_DIM; ++j) {
	//		tmp = RTE->state_trajectories_host[idx];
	//		std::cout << tmp << "    ";
	//		idx++;
	//	};
	//	sim_idx++;
	//	std::cout << "" << std::endl;
	//}

	// Assert that there are no NaN's
	for (int i = 0; i < num_rollouts; ++i) {
		for (int j = 0; j < DubinDynamics::STATE_DIM; ++j) {
			ASSERT_FALSE(isnan(
				RTE->state_trajectories_host[num_timesteps * DubinDynamics::STATE_DIM * i // Rollout
				+ (num_timesteps - 1) * DubinDynamics::STATE_DIM //Timestep
				+ j])) << "Rollout: " << i << ", State: " << j << std::endl; // State
		}
	}
}