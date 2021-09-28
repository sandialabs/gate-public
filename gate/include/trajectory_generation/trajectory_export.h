#ifndef TRAJECTORY_EXPORT_H
#define TRAJECTORY_EXPORT_H

/* 
* These functions will be used to export trajectories in a more compact manner.
* 
* 
* 
*/


#include <vector>
#include <string>
//#include <filesystem> // C++17 only
#include <direct.h>
#include <fstream>
#include <ctime>
#include <iostream>
#include <chrono>

//namespace fs = std::filesystem;

namespace GATE_internal {

	std::string project_root = "../../../gate";

	// TODO function to get the current datetime for naming
	// TODO function to output the parameters (num_timesteps, etc) into a file

	// Function to make sure the folder where we are saving exists
	std::string create_save_location(std::string system_name, std::string project_root_path = project_root) {
		// The saving portion will go into gate/trajectories/[system_type]
		std::string trajectory_path = project_root_path + "/trajectories/" + system_name + "/";
		//fs::create_directories(trajectory_path);
		mkdir(trajectory_path.data());
		return trajectory_path;
	}

	std::string get_datetime() {
		std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
		std::time_t start_time = std::chrono::system_clock::to_time_t(now);
		char timedisplay[100];
		tm buf;
		errno_t err = localtime_s(&buf, &start_time);
		std::strftime(timedisplay, sizeof(timedisplay), "_%Y_%m_%d__%H_%M_%S.csv", &buf);
		return std::string(timedisplay);
	}


	void save_traj_bundle(int state_dim, int num_timesteps, int num_trajectories, std::vector<float> x_traj, std::string system_name) {
		// let the first 2 values be state_dim, traj_len
		auto start = std::chrono::system_clock::now();
		auto trajectory_path = create_save_location(system_name);
		auto datetime = get_datetime();
		std::ofstream traj_file(trajectory_path + system_name + datetime);
		traj_file << state_dim << ", " << num_timesteps << ", " << num_trajectories << ", ";
		for (int i = 0; i < state_dim * num_timesteps * num_trajectories; ++i) {
			traj_file << x_traj[i];
			if (i == state_dim * num_timesteps * num_trajectories) {
				traj_file << "\n";
			}
			else {
				traj_file << ", ";
			}
		}
		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Trajectory Export Time: " << elapsed.count() / 1000.0 << " seconds." << std::endl;
	}

}


#endif