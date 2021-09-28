#include <trajectory_export.h>
#include <trajectory_generation/GATE.cuh>
#include <dynamics/mass_spring_damper/mass_spring_damper_dynamics.cuh>
#include <controllers/mass_spring_damper/mass_spring_damper_PID.cuh>
#include <controllers/mass_spring_damper/mass_spring_damper_LQR.cuh>
#include <perturbations/mass_spring_damper/mass_spring_damper_perturbations.cuh>
#include <dynamics/integrators_eigen.cuh>
#include <memory>
#include <iostream>

int main() {
	// NOTE: When saving trajectories we assume that the binary is located in
	// */gate-public/build/[BUILD-CONFIGURATION]/bin
	// [BUILD-CONFIGURATION] examples are Debug, Release, etc.
	// This assumption will be true if we use CMake to setup the build system for this project.
	std::string system_name = "spring_mass_damper";

	// simulation timestep
	float dt = 0.01;

	// Setup rollouts
	const int num_timesteps = 1000; // 2500;
	const int num_rollouts = 500; // 500;
	const int bdim_x = 1;

	// set params
	std::shared_ptr<MsdDynamics> dyn = std::make_shared<MsdDynamics>();

	// PID Controller
	std::shared_ptr<MsdPID> ctrl = std::make_shared<MsdPID>();
	auto default_guid_params = ctrl->getGuidanceParams();
	default_guid_params.dt = dt;
	default_guid_params.p_gain = 1;
	default_guid_params.i_gain = 0.3f;


	// LQR Controller
	//std::shared_ptr<MsdLQR> ctrl = std::make_shared<MsdLQR>();
	//auto default_guid_params = ctrl->getGuidanceParams();
	//default_guid_params.dt = dt;
	//default_guid_params.setNumTimesteps(num_timesteps);


	auto default_dyn_params = ctrl->getDynamicsParams();
	ctrl->setGuidanceParams(default_guid_params);
	ctrl->setDynamicsParams(default_dyn_params);
	dyn->setParams(default_dyn_params);
	//****Include if doing LQR controller****
	//ctrl->computeP();

	// perturbations
	MsdPertParams::state_array x0_mean, x0_std;
	MsdPertParams::control_array u_std;
	x0_mean << 1.f, 0.1f;
	x0_std << 0.01f, 0.01f;
	u_std << .01;
	MsdPertParams params = MsdPertParams(x0_mean, x0_std, u_std);
	std::shared_ptr<MsdPert<num_timesteps, num_rollouts>> pert = std::make_shared<MsdPert<num_timesteps, num_rollouts>>(params);
	pert->initializeX0andControlPerturbations();
	pert->initPerturbations();

	// Create the GATE object
	typedef GATE<MsdDynamics, MsdPID, MsdPert<num_timesteps, num_rollouts>, num_timesteps, num_rollouts, bdim_x> MSD_ROTE;
	std::shared_ptr<MSD_ROTE> RTE = std::make_shared<MSD_ROTE>(dyn.get(), ctrl.get(), pert.get(), x0_mean, dt);

	RTE->computeTrajectories();

	GATE_internal::save_traj_bundle(MsdDynamics::STATE_DIM, num_timesteps, num_rollouts, RTE->state_trajectories_host, system_name);

	return 0;
}