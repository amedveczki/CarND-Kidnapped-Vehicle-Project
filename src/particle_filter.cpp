/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <climits>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  // This should likely be based on std + desired accuracy
  num_particles = 100;


  std::normal_distribution<double> xdist{x, std[0]};
  std::normal_distribution<double> ydist{y, std[1]};
  std::normal_distribution<double> tdist{theta, std[2]};

  for (int i = 0; i < num_particles; ++i)
      particles.push_back({i, xdist(gen), ydist(gen), tdist(gen), 1});
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate)
{
  std::normal_distribution<double> xdist{0, std_pos[0]};
  std::normal_distribution<double> ydist{0, std_pos[1]};
  std::normal_distribution<double> tdist{0, std_pos[2]};
    for (auto&& p : particles)
    {
        p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta))
            + xdist(gen);
        p.y += velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t))
            + ydist(gen);
        p.theta += yaw_rate * delta_t
            + tdist(gen);
    }
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    for (auto& obs : observations)
    {
        double nearestDist = std::numeric_limits<double>::max();
        const LandmarkObs * nearest = nullptr;

        for (const auto& pred : predicted)
        {
            const double dx = pred.x - obs.x;
            const double dy = pred.y - obs.y;

            // No sqrt, hope it's faster
            const double dist = dx*dx + dy*dy;

            if (dist < nearestDist)
            {
                nearestDist = dist;
                nearest = &pred;
            }
        }

        assert(nearest);
        obs = *nearest;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

    const double maxDist = sensor_range*sensor_range;
    assert(maxDist > sensor_range);
 

    for (auto& p : particles)
    {

        vector<LandmarkObs> predicted;

        // Translate+rotate landmark positions so observations and predicted
        // can be compared
        for (const auto& lm : map_landmarks.landmark_list)
        {
            double vx = p.x + (cos(p.theta)*lm.x_f - sin(p.theta)*lm.y_f);
            double vy = p.y + (sin(p.theta)*lm.x_f + cos(p.theta)*lm.y_f);

            double xdiff = vx-p.x;
            double ydiff = vx-p.x;

            if (xdiff*xdiff + ydiff*ydiff < maxDist)
                predicted.push_back({lm.id_i, vx, vy});
        }

        assert(observations.size() == predicted.size());

        // pair observations to predicted landmarks
        vector<LandmarkObs> matched_observations(observations);
        dataAssociation(predicted, matched_observations);

        p.weight = 1;

        // Update weights based on this
        for (size_t i = 0; i < observations.size(); ++i)
        {
            const auto& obs = observations[i];
            const auto& pred = predicted[i];
            const auto dx = obs.x - pred.x;
            const auto dy = obs.y - pred.y;
            const auto& ox = std_landmark[0];
            const auto& oy = std_landmark[1];

            p.weight *= (1/(2*M_PI*ox*oy)) * exp(-(dx*dx/(2*ox*ox) + dy*dy/(2*oy*oy)));
        }
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}