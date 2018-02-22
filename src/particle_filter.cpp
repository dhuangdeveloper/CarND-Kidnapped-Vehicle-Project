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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  num_particles = 100;
  default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);	
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);  
  particles.clear();
  //weights.clear();
  for (int i =0; i< num_particles; ++i){
      particles.push_back({i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0, std::vector<int>(), std::vector<double>(), std::vector<double>()});
      //weights.push_back(1.0);
  }
	is_initialized = true;
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
  default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);	
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);   
  for (int i =0; i< num_particles; ++i){
    if (abs(yaw_rate)>=0.01){
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + dist_x(gen);
      particles[i].y += velocity / yaw_rate * (-cos(particles[i].theta + yaw_rate * delta_t) + cos(particles[i].theta)) + dist_y(gen);
      particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
    } else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
      particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
      particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
    }
  }
  
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
  for (std::vector<LandmarkObs>::iterator it = observations.begin() ; it != observations.end(); ++it){
    double min_dist = dist((*predicted.cbegin()).x, (*predicted.cbegin()).y, (*it).x, (*it).y);
    double id_min_dist = (*predicted.cbegin()).id;
    for (std::vector<LandmarkObs>::iterator it_predicted = predicted.begin() ; it_predicted != predicted.end(); ++it_predicted){
      double current_dist = dist((*it_predicted).x, (*it_predicted).y, (*it).x, (*it).y);
      if (current_dist < min_dist){
        id_min_dist = (*it_predicted).id;
        min_dist = current_dist;
      }        
    }
    (*it).id = id_min_dist;
  }   
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  // First calculate transform landmarks to predicted observations
  // convert map_landmarks to landmarkobs

  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];  
  for (std::vector<Particle>::iterator it = particles.begin() ; it != particles.end(); ++it){
    std::vector<LandmarkObs> predicted_map;
    for (std::vector<Map::single_landmark_s>::const_iterator it_lm = map_landmarks.landmark_list.begin() ; it_lm != map_landmarks.landmark_list.end(); ++it_lm){
      if (dist((*it).x, (*it).y, (*it_lm).x_f, (*it_lm).y_f) <= sensor_range){
        predicted_map.push_back({(*it_lm).id_i, (*it_lm).x_f, (*it_lm).y_f});
      }
    }    
    std::vector<LandmarkObs> observation_map;
    for (std::vector<LandmarkObs>::const_iterator it_obs = observations.begin() ; it_obs != observations.end(); ++it_obs){
      double xm = cos((*it).theta) * (*it_obs).x - sin((*it).theta)* (*it_obs).y + (*it).x;
      double ym = sin((*it).theta) * (*it_obs).x + cos((*it).theta)* (*it_obs).y + (*it).y;
      observation_map.push_back({-1, xm, ym});
    }
    dataAssociation(predicted_map, observation_map);
    // calculate probability distribution
    double log_probability = 0;
    for (std::vector<LandmarkObs>::const_iterator it_obs = observation_map.begin() ; it_obs != observation_map.end(); ++it_obs){
      for (std::vector<LandmarkObs>::const_iterator it_pre = predicted_map.begin() ; it_pre != predicted_map.end(); ++it_pre){
        if ((*it_pre).id == (*it_obs).id){
          log_probability += log(1.0/2/M_PI/sigma_x/sigma_y) -(pow((*it_pre).x-(*it_obs).x, 2) / 2/sigma_x/sigma_x + pow((*it_pre).y-(*it_obs).y, 2) / 2/sigma_y/sigma_y);          
        }
      }
    }
    (*it).weight *= exp(log_probability);       
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  //std::vector<double> weights;
  double max_weight = 0;
  for (std::vector<Particle>::iterator it = particles.begin() ; it != particles.end(); ++it){
    if ((*it).weight> max_weight){
      max_weight=(*it).weight;      
    }
    //weights.push_back((*it).weight);
  }
  
  std::default_random_engine gen;
  
  std::uniform_real_distribution<double> dist_beta_delta(0, 2*max_weight);
  std::vector<Particle> new_particles;
  //weights.clear();
  double beta = 0;
  int index = 0;
  for (int i=0; i< num_particles; ++i){
    //new_particles.push_back(particles[dist_d(gen)]);  
    beta += dist_beta_delta(gen);
    while (particles[index].weight< beta){
      beta -= particles[index].weight;
      index = (index+1) % num_particles;
    }
    new_particles.push_back({i, particles[index].x, particles[index].y, particles[index].theta, 1.0, std::vector<int>(), std::vector<double>(), std::vector<double>()});
    //weights.push_back(1.0);
  }
  particles.clear();
  particles = new_particles;
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
