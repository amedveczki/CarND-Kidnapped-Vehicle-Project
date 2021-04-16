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
  // This should likely be based on std + desired accuracy
  num_particles = 50;

  std::normal_distribution<double> xdist{x, std[0]};
  std::normal_distribution<double> ydist{y, std[1]};
  std::normal_distribution<double> tdist{theta, std[2]};

  std::cout << "Init, gps: " << x << "," << y << " theta " << theta << std::endl;

  for (int i = 0; i < num_particles; ++i)
  {
      particles.push_back({i, xdist(gen), ydist(gen), tdist(gen), 1});
      const auto& p = particles.back();
      std::cout << "Init, particle #" << i << " " << p.x << "," << p.y << " theta " << p.theta << std::endl;
  }

  is_initialized = true;
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
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

    // Associate transformed observations with landmark IDs based on distance
    for (auto& obs : observations)
    {
        double nearestDist = std::numeric_limits<double>::max();
        int nearest = -1;

        for (size_t i = 0; i < predicted.size(); ++i)
        {
            const double dx = predicted[i].x - obs.x;
            const double dy = predicted[i].y - obs.y;

            // No sqrt, hope it's faster
            const double dist = dx*dx + dy*dy;

            if (dist < nearestDist)
            {
                nearestDist = dist;
                nearest = i;
            }
        }

        obs.id = nearest;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
    const double maxDist = sensor_range*sensor_range;
    assert(maxDist > sensor_range);
 
    for (auto& p : particles)
    {
        vector<LandmarkObs> transformed;
        vector<LandmarkObs> predicted;

        // Check which landmarks are in range
        for (const auto& lm : map_landmarks.landmark_list)
        {
            double xdiff = lm.x_f-p.x;
            double ydiff = lm.y_f-p.y;

            if (xdiff*xdiff + ydiff*ydiff < maxDist)
                predicted.push_back({lm.id_i, lm.x_f, lm.y_f});
        }

        // transform observed landmarks to map space
        for (const auto& obs : observations)
        {
            double vx = p.x + (cos(p.theta)*obs.x - sin(p.theta)*obs.y);
            double vy = p.y + (sin(p.theta)*obs.x + cos(p.theta)*obs.y);

            transformed.push_back({obs.id, vx, vy});
        }

        // pair observations to predicted landmarks
        dataAssociation(predicted, transformed);

        p.weight = 1;

        // Update weights based on the differences
        for (size_t i = 0; i < transformed.size(); ++i)
        {
            const auto& obs = transformed[i];
            const auto& pred = predicted[obs.id];
            const auto dx = obs.x - pred.x;
            const auto dy = obs.y - pred.y;
            const auto& ox = std_landmark[0];
            const auto& oy = std_landmark[1];

            p.weight *= (1/(2*M_PI*ox*oy)) * exp(-(dx*dx/(2*ox*ox) + dy*dy/(2*oy*oy)));
        }
    }
}

void ParticleFilter::resample() {
    // Vector of weights of all particles
    std::vector<double> weights; 
    std::transform(particles.begin(), particles.end(), std::back_inserter(weights),
            [](const Particle& p) { return p.weight; });

    std::discrete_distribution<> disc(weights.begin(), weights.end());

    std::vector<Particle> new_particles;

    for (size_t i = 0; i < particles.size(); ++i)
        new_particles.push_back(particles[disc(gen)]);

    particles = std::move(new_particles);
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
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
