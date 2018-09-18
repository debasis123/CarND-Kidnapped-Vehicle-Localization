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
#include <cassert>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  std::default_random_engine gen;

  num_particles = 100;
  // Create a normal (Gaussian) distribution for x, y, theta around corresponding means
  std::normal_distribution<> ndist_x(x, std[0]),
                             ndist_y(y, std[1]),
                             ndist_theta(theta, std[2]);
  
  for (size_t i = 0; i != num_particles; ++i) {
    // Sample from the above normal distributions 
    Particle p;
    p.id     = i;
    p.x      = ndist_x(gen);
    p.y      = ndist_y(gen);
    p.theta  = ndist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
    weights.push_back(p.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  std::default_random_engine gen;

  ////////////////////////
  // DETERMINISTIC PART //
  ////////////////////////
  
  // bicycle model
  for (auto& p : particles) {
    if (std::fabs(yaw_rate) > 0.00001) {
      p.x     += (velocity/yaw_rate) * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y     += (velocity/yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      p.theta += yaw_rate * delta_t;
    }
    else {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }

    /////////////////////
    // STOCHASTIC PART //
    /////////////////////

    // Create a normal (Gaussian) distribution for x, y, theta around corresponding means
    std::normal_distribution<> ndist_x(p.x, std_pos[0]),
                               ndist_y(p.y, std_pos[1]),
                               ndist_theta(p.theta, std_pos[2]);
    
    // random Gaussian noise, sample from above normal distributions 
    p.x     = ndist_x(gen);
    p.y     = ndist_y(gen);
    p.theta = ndist_theta(gen);
  }
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

  // for each particle, we have to update all the observations
  double sum_weights = 0.0;
  for (auto& p: particles) {

    ////////////////////
    // TRANSFORMATION //
    ////////////////////

    std::vector<LandmarkObs> trans_observations;
    for (const auto& o: observations) {
      LandmarkObs trans_obs;
      trans_obs.x = p.x + o.x*cos(p.theta) - o.y*sin(p.theta);
      trans_obs.y = p.y + o.x*sin(p.theta) + o.y*cos(p.theta);
      trans_observations.push_back(trans_obs);
    }

    //////////////////////////////
    // ASSOCIATION TO LANDMARKS //
    //////////////////////////////

    for (auto& tr_obs : trans_observations) {
      double min_dist = std::numeric_limits<const double>::infinity();
      int best_id = -1;

      for (const auto& lm : map_landmarks.landmark_list) {
        double d = dist(tr_obs.x, tr_obs.y, lm.x_f, lm.y_f);

        if (fabs(p.x - lm.x_f) <= sensor_range
            && fabs(p.y - lm.y_f) <= sensor_range
            && d < sensor_range * sqrt(2)
            && d < min_dist)
        {
          best_id = lm.id_i;
          min_dist = d;
        }
      }

      assert(best_id <= 42 && "map has a max id of 42");
      
      tr_obs.id = best_id;
    }


    ////////////////////
    // UPDATE WEIGHTS //
    ////////////////////
    
    p.weight = 1.0;
    const double sig_x    = std_landmark[0];
    const double sig_y    = std_landmark[1];
    const double den_term = 1.0 / (2.0 *M_PI * sig_x*sig_y);

    for (auto& tr_obs : trans_observations) {
      const int asso_id = tr_obs.id;
      const double x        = tr_obs.x;
      const double y        = tr_obs.y;

      double norm_prob = 1.0;
      for (auto& lm : map_landmarks.landmark_list) {
        if (lm.id_i == asso_id) {
          const double mu_x = lm.x_f;
          const double mu_y = lm.y_f;
          const double exp_term = exp( 
                                  -pow(x-mu_x, 2) / (2*sig_x*sig_x) 
                                  -pow(y-mu_y, 2) / (2*sig_y*sig_y));

          norm_prob = den_term * exp_term;
          p.weight *= norm_prob;
        }
      }
    }
    sum_weights += p.weight;
  }

  // normalizing the particle weights
  for (size_t i = 0; i!= particles.size(); ++i) {
    particles[i].weight /= sum_weights;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::default_random_engine gen;

  std::discrete_distribution<> dist(weights.begin(), weights.end());

  std::vector<Particle> resampled_particles;
  for (size_t i = 0; i != num_particles; ++i) {
    const int index = dist(gen);
    resampled_particles.push_back(particles.at(index));
  }

  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

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
