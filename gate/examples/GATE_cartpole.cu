#include <trajectory_export.h>
#include <trajectory_generation/GATE.cuh>
#include <dynamics/cartpole/cartpole_dynamics.cuh>
#include <controllers/cartpole/cartpole_LQR.cuh>
#include <perturbations/cartpole/cartpole_perturbations.cuh>
#include <dynamics/integrators_eigen.cuh>
#include <memory>
#include <iostream>

int main() {
	// NOTE: When saving trajectories we assume that the binary is located in
	// */gate-public/build/[BUILD-CONFIGURATION]/bin
	// [BUILD-CONFIGURATION] examples are Debug, Release, etc.
	// This assumption will be true if we use CMake to setup the build system for this project.
	std::string system_name = "cartpole";

	// simulation timestep
	float dt = 0.01;

	// Setup rollouts
	const int num_timesteps = 1500;
	const int num_rollouts = 500; // 500;
	const int bdim_x = 1;

	// set params
	std::shared_ptr<CartpoleDynamics> dyn = std::make_shared<CartpoleDynamics>();


	// LQR Controller
	std::shared_ptr<CartpoleLQR> ctrl = std::make_shared<CartpoleLQR>();
	auto default_guid_params = ctrl->getGuidanceParams();
	//default_guid_params.dt = dt; //todo is this necessary?
	//default_guid_params.setNumTimesteps(num_timesteps);
	
	
	auto default_dyn_params = ctrl->getDynamicsParams();
	ctrl->setGuidanceParams(default_guid_params);
	ctrl->setDynamicsParams(default_dyn_params);
	dyn->setParams(default_dyn_params);
	
	// perturbations
	CartpolePertParams::state_array x0_mean, x0_std;
	CartpolePertParams::control_array u_std;
	x0_mean << -1.f, 0.0f, 0.0f, 0.0f;
	x0_std << 0.0f, 0.0f, 0.0f, 0.0f;
	u_std << .1;
	CartpolePertParams params = CartpolePertParams(x0_mean, x0_std, u_std);
	std::shared_ptr<CartpolePert<num_timesteps, num_rollouts>> pert = std::make_shared<CartpolePert<num_timesteps, num_rollouts>>(params);
	pert->initializeX0andControlPerturbations();
	pert->initPerturbations();

	
	// Create the GATE object
	typedef GATE<CartpoleDynamics, CartpoleLQR, CartpolePert<num_timesteps, num_rollouts>, num_timesteps, num_rollouts, bdim_x> Cartpole_ROTE;
	std::shared_ptr<Cartpole_ROTE> RTE = std::make_shared<Cartpole_ROTE>(dyn.get(), ctrl.get(), pert.get(), x0_mean, dt);

	RTE->computeTrajectories();
	
	GATE_internal::save_traj_bundle(CartpoleDynamics::STATE_DIM, num_timesteps, num_rollouts, RTE->state_trajectories_host, system_name);

	
	return 0;
}