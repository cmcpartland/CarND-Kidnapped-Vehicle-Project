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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	
	const double std_x = std[0];
	const double std_y = std[1];
	const double std_theta = std[2];
	
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	
	num_particles = 100;
	particles.reserve(num_particles);
	
	for (int i = 0; i < num_particles; i++) {
		// Generate x, y, and theta samples from a Gaussian distribution
		// Instantiate a Particle p with the sampled x,y,theta values, weight=1, and id=index number
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		weights.push_back(p.weight);
		
		// Add the generated Particle to the particles vectors
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	cout << "Predicting..." << endl;
	default_random_engine gen;
	
	const double std_x_meas = std_pos[0];
	const double std_y_meas = std_pos[1];
	const double std_theta_meas = std_pos[2];
	
	for (int i = 0; i < num_particles; i++) {
		Particle current_particle = particles[i];
		const double x_prev = current_particle.x;
		const double y_prev = current_particle.y;
		const double theta_prev = current_particle.theta;
	
		// Predict particle pose based on process model
		if (yaw_rate == 0) {
			yaw_rate = 0.0001;
		}
		const double v_over_thetadot = velocity/yaw_rate;
		const double theta_dot_times_delta_t = yaw_rate*delta_t;
		
		const double x_new = x_prev*v_over_thetadot*(sin(theta_prev + theta_dot_times_delta_t) - sin(theta_prev));
		const double y_new = y_prev*v_over_thetadot*(cos(theta_prev) - cos(theta_prev + theta_dot_times_delta_t));
		const double theta_new = theta_prev + theta_dot_times_delta_t;
		
		// Create Gaussian distributions to sample from to include process noise
		normal_distribution<double> dist_x(x_new, std_x_meas);
		normal_distribution<double> dist_y(y_new, std_y_meas);
		normal_distribution<double> dist_theta(theta_new, std_theta_meas);
		
		// Update the particle with the predicted pose
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// Predicted measurements come from where the landmarks would be predicted to be assuming that the particle position is correct.
	for (int i = 0; i < observations.size(); i++) {
		//cout << "Observation " << i << " being associated" << endl;
		double min_distance = 0;
		int associated_id = 0;
		for (int j = 0; j < predicted.size(); j++) {
			// const double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			const double diff_x = observations[i].x - predicted[j].x;
			const double diff_y = observations[i].y - predicted[j].y;
			const double distance = sqrt(pow(diff_x,2) + pow(diff_y,2));
			//cout << "Distance: " << distance << endl;
			if (j == 0) {
				min_distance = distance;
			}
			if (distance <= min_distance) {
				associated_id = predicted[j].id; // associate the predicted measurement with the closest observed measurement
				min_distance = distance;
			}
		}
		observations[i].id = associated_id; // Associate closest observation
		//cout << "Observation " << i << " succcessfully associated" << endl;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs>& observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	const double sigma_x = std_landmark[0];
	const double sigma_y = std_landmark[1];
	const double mv_constant = 1.0/(2*M_PI*sigma_x*sigma_y);
	
	// 1. Move through each particle to update the weight
	for (int i = 0; i < num_particles; i++) {
		// Grab the current particle's pose, which is in the map frame
		Particle current_particle = particles[i];
		const double x_p = current_particle.x;
		const double y_p = current_particle.y;
		const double theta_p = current_particle.theta;
		
		std::vector<LandmarkObs> predicted_landmarks_p; // to be filled with predicted landmarks in particle frame
		
		//cout << "Step 1 finished." << endl;
		
		// 2. Transform each landmark from map frame to particle frame
		for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			Map::single_landmark_s map_landmark = map_landmarks.landmark_list[j];
			
			const double x_landmark = map_landmark.x_f; 
			const double y_landmark = map_landmark.y_f;
			
			// If landmark is within particle sensor range, add it to the list of landmarks
			if ((abs (x_p - x_landmark) <= sensor_range) && (abs (y_p - y_landmark) <= sensor_range)) {
			
				// x-coordinate of landmark j in particle frame
				const double x_landmark_p = cos(theta_p)*x_landmark + sin(theta_p)*y_landmark - x_p*cos(theta_p) - y_p*sin(theta_p); 
				// y-coordinate of landmark j in particle frame
				const double y_landmark_p = -sin(theta_p)*x_landmark + cos(theta_p)*y_landmark + x_p*sin(theta_p) - y_p*cos(theta_p); 
				
				LandmarkObs predicted_landmark_p;
				predicted_landmark_p.id = map_landmarks.landmark_list[j].id_i;
				predicted_landmark_p.x = x_landmark_p;
				predicted_landmark_p.y = y_landmark_p;
				
				predicted_landmarks_p.push_back(predicted_landmark_p); // Add landmark (in particle frame) to vector
			}
		}
		// 3. Use nearest neighbor to find which observations are associated with which landmark
		dataAssociation(predicted_landmarks_p, observations);
		
		// Go through each observation and 
		double P = 1.0;
		for (int k = 0; k < observations.size(); k++) {
			const double x_diff = observations[k].x - predicted_landmarks_p[observations[k].id].x;
			//cout << "got x_diff" << endl;
			const double y_diff = observations[k].y - predicted_landmarks_p[observations[k].id].y;
			//cout << "got y_diff" << endl;
			P *= mv_constant*exp(-(0.5*(pow(x_diff/sigma_x, 2) + pow(y_diff/sigma_y, 2))));
			//cout << "For observation " << k << " multiplier is : " << P << endl;
		}
		//cout << " P is calculated." << endl;
		//cout << "Weights of i = " << weights[i] << endl;
		particles[i].weight = P;
		weights[i] = P;
		//cout << "Step 3 finished. Weights updated." << endl;
	}			
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<> d(weights.begin(), weights.end());
	map<int, int> m;
	std::vector<Particle> temp;
	for (int i = 0; i < num_particles; i++) {
		++m[d(gen)];
	}
	
	for (auto p : m) {
		for (int j = 0; j < p.second; j++) {
			temp.push_back(particles[p.first]);
		}
	}
	particles = temp;	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
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
