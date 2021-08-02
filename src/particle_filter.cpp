/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>
#include <random>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	default_random_engine gen;
	
	// Grab the standard deviations for x, y, and theta values...
	const double std_x = std[0];
	const double std_y = std[1];
	const double std_theta = std[2];
	
	// and use them to create normal Gaussian distributions centered around the x,y,theta values.
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	
	// Start with 100 particles.
	num_particles = 100;
	particles.reserve(num_particles);
	
	for (int i = 0; i < num_particles; i++) {
		// Generate x, y, and theta samples from a Gaussian distribution.
		// Instantiate a Particle p with the sampled x,y,theta values, weight=1, and id=index number...
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		weights.push_back(1.0);
		
		// then add the generated Particle to the particles vectors.
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	
	default_random_engine gen;
	
	// Grab the standard deviation values for the prediction step.
	const double std_x_meas = std_pos[0];
	const double std_y_meas = std_pos[1];
	const double std_theta_meas = std_pos[2];
	
	// Go through each particle...
	for (int i = 0; i < num_particles; i++) {
		// and collect the x,y,theta values and mark them as the 'previous' values.
		Particle current_particle = particles[i];
		const double x_prev = current_particle.x;
		const double y_prev = current_particle.y;
		const double theta_prev = current_particle.theta;
	
		// Set aside a couple commonly used constants for efficiency.		
		const double v_over_thetadot = velocity/yaw_rate;
		const double theta_dot_times_delta_t = yaw_rate*delta_t;
		
		// Predict particle pose based on process model. Instantiate new values for x,y,theta. 
		double x_new, y_new, theta_new;
		
		// Consider the case where the yaw rate is close to zero. If it's close to zero, assume the particle continues on a straight line.
		if (abs(yaw_rate) < 0.00001) {
			x_new = x_prev + velocity*delta_t*cos(theta_prev);
			y_new = y_prev + velocity*delta_t*sin(theta_prev);
			theta_new = theta_prev;
		}
		// Otherwise, if the yaw rate value is appreciably large, then use the motion model to predict the new x,y,theta values. 
		else {
			x_new = x_prev + v_over_thetadot*(sin(theta_prev + theta_dot_times_delta_t) - sin(theta_prev));
			y_new = y_prev + v_over_thetadot*(cos(theta_prev) - cos(theta_prev + theta_dot_times_delta_t));
			theta_new = theta_prev + theta_dot_times_delta_t;
		}
		// To introduce process noise, create Gaussian distributions centered around new x,y,theta values from which to sample. 
		normal_distribution<double> dist_x(x_new, std_x_meas);
		normal_distribution<double> dist_y(y_new, std_y_meas);
		normal_distribution<double> dist_theta(theta_new, std_theta_meas);
		
		// Finally, update the particle with the newly predicted pose. 
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//  Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.

	// For this step, both the predicted landmarks and the observed landmarks are in the MAP frame.
	// The predicted landmarks are the landmarks that the particle should be able to sense (i.e. within range). 
	
	// Go through each observed landmark...
	for (int i = 0; i < observations.size(); i++) {
		double min_distance = 0.0; // > initialize the minimum distance between observed landmark and predicted landmark as 0.
		int associated_id = -1;	   // > initialize the associated landmark id as -1 (implausible)
		
		// and for each observed landmark, loop through all the known map landmarks and search for the closest one. 
		for (int j = 0; j < predicted.size(); j++) {
			const double diff_x = observations[i].x - predicted[j].x;
			const double diff_y = observations[i].y - predicted[j].y;
			const double distance = sqrt(pow(diff_x,2) + pow(diff_y,2));
			if (j == 0) { 	// > this is only here to correctly start the search process
				min_distance = distance;
			}
			if (distance <= min_distance) {
				associated_id = predicted[j].id; // > associate the predicted measurement with the closest observed measurement
				min_distance = distance;
			}
		}
		// Finally, associate the observation with the closest known landmark. 
		observations[i].id = associated_id; 
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs>& observations, const Map &map_landmarks) {
	// Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. The particles are located
	//   according to the MAP'S coordinate system. The locations will be transformed between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	// Grab the standard deviation values for the update step. 
	const double sigma_x = std_landmark[0];
	const double sigma_y = std_landmark[1];

	const double mv_constant = 1.0/(2*M_PI*sigma_x*sigma_y); // > constant to be used later
	
	// Loop through each particle...
	for (int i = 0; i < num_particles; i++) {
		// and grab the current particle's pose, which is in the map frame. 
		Particle current_particle = particles[i];
		const double x_p = current_particle.x;
		const double y_p = current_particle.y;
		const double theta_p = current_particle.theta;
		
		std::vector<LandmarkObs> predicted_landmarks_m; // > to be filled with predicted landmarks in map frame
		
		// Next, loop through the map landmarks...
		for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			Map::single_landmark_s map_landmark = map_landmarks.landmark_list[j];
			
			const double x_landmark = map_landmark.x_f; 
			const double y_landmark = map_landmark.y_f;
			const int id_landmark = map_landmark.id_i;
			
			// and only consider the ones that are within the particle's sensor range.
			if ((abs (x_p - x_landmark) <= sensor_range) && (abs (y_p - y_landmark) <= sensor_range)) {			
				
				LandmarkObs predicted_landmark_m;
				predicted_landmark_m.x = x_landmark;
				predicted_landmark_m.y = y_landmark;
				predicted_landmark_m.id = id_landmark;
				
				predicted_landmarks_m.push_back(predicted_landmark_m);
			}
		}
		
		// Now, loop through the observations...
		std::vector<LandmarkObs> observations_m;
		for (int k = 0; k < observations.size(); k++) {
			// and first extract that x,y position of the observation in the particle frame. 
			const double x_obs_p = observations[k].x;
			const double y_obs_p = observations[k].y;
			const int id_obs = observations[k].id;
			
			// Then transform the x,y position of the observation from the particle frame to the map frame. 
			const double x_obs_m = cos(theta_p)*x_obs_p - sin(theta_p)*y_obs_p + x_p;
			const double y_obs_m = sin(theta_p)*x_obs_p + cos(theta_p)*y_obs_p + y_p;
			
			// Now add the transformed observation to the list. 
			LandmarkObs observation_m;
			observation_m.x = x_obs_m;
			observation_m.y = y_obs_m;
			observation_m.id = id_obs;
			observations_m.push_back(observation_m);
		}
		
		// Use nearest neighbor to find which observations are associated with which landmark. 
		dataAssociation(predicted_landmarks_m, observations_m);
		
		// Now that the observations are associated with landmarks, loop through each observation...
		double P = 1.0;
		for (int q = 0; q < observations_m.size(); q++) {
			const int id_associated_landmark = observations_m[q].id;
			double x_diff = 0.0; // > will hold the difference in x coordinate between the observed landmark and the associated map landmark
			double y_diff = 0.0; // > same for diffrence in y coordinate
			
			// and find its associated landmark. Then calculate the distance between the observed landmark and the associated map landmark. 
			for (int r = 0; r < predicted_landmarks_m.size(); r++) {
				if (predicted_landmarks_m[r].id == id_associated_landmark) {
					x_diff -= predicted_landmarks_m[r].x;
					y_diff -= predicted_landmarks_m[r].y;
				}
			}
			x_diff += observations_m[q].x;
			y_diff += observations_m[q].y;
			
			// Calculate the Multivariate Gaussian probability density, which is calculating the probability density
			// of observing the observation given the known landmark positions. 
			P *= mv_constant*exp(-(0.5*(pow(x_diff/sigma_x, 2) + pow(y_diff/sigma_y, 2))));
		}
		// Finally, update the particle weight and the corresponding entry in the weights list.
		particles[i].weight = P;
		weights[i] = P;
	}			
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	default_random_engine gen;
	std::vector<Particle> temp; // a temporary list of Particles
	
	// Pick out the max weight value from the weights vector. 
	double max_weight = *max_element(weights.begin(), weights.end());
	
	// Choose an index at random.
	int index = rand() % num_particles;
	
	uniform_real_distribution<> dist_value(0,1);
	double beta = 0.0;
	
	// Use a probability wheel spin to choose particles from the list with replacement. 
	for (int i = 0; i < num_particles; i++) {
		beta += dist_value(gen)*2.0*max_weight;
		while (weights[index] < beta) {
			beta -= weights[index];
			index = (index+1) % num_particles;
		}
		temp.push_back(particles[index]);
	}
	// Once all the selected particles have been added to temp, make 'particles' point to the new vector. 
	particles = temp;	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
