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
  num_particles = 10;


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
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate)
{
  std::normal_distribution<double> xdist{0, std_pos[0]};
  std::normal_distribution<double> ydist{0, std_pos[1]};
  std::normal_distribution<double> tdist{0, std_pos[2]};
        //std::cout << "Prediction t: " << delta_t << " velocity " << velocity << " yaw_rate " << yaw_rate << std::endl;;
        int i = 0;
    for (auto&& p : particles)
    {
        p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta))
            + xdist(gen);
        p.y += velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t))
            + ydist(gen);
        p.theta += yaw_rate * delta_t
            + tdist(gen);

        //std::cout << " Particle " << i++ << " -> " << p.x << "," << p.y << " theta " << p.theta << std::endl;
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
    /*
    std::cout << "dataAssociation - predicted" << std::endl;
    for (size_t i = 0; i < predicted.size(); ++i)
        std::cout << i << ". pred: " << predicted[i].x << "," << predicted[i].y << std::endl;
    for (size_t i = 0; i < observations.size(); ++i)
        std::cout << i << ". obs: " << observations[i].x << "," << observations[i].y << std::endl;

    std::cout << "Observations..." << std::endl;
    */

    vector<LandmarkObs> unsortedObservations(std::move(observations));
    for (auto& pred : predicted)
    {
        double nearestDist = std::numeric_limits<double>::max();
        const LandmarkObs * nearest = nullptr;

        for (const auto& obs : unsortedObservations)
        {
            const double dx = pred.x - obs.x;
            const double dy = pred.y - obs.y;

            // No sqrt, hope it's faster
            const double dist = dx*dx + dy*dy;

            if (dist < nearestDist)
            {
                nearestDist = dist;
                nearest = &obs;
            }
        }

        assert(nearest);

        //std::cout << "observed orig " << nearest->x << "," << nearest->y << " => predicted " << pred.x << "," << pred.y << std::endl;
        observations.push_back(*nearest);
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
 
    vector<LandmarkObs> transformed;

    for (auto& p : particles)
    {
        vector<LandmarkObs> predicted;

        // Check which landmarks are in range
        for (const auto& lm : map_landmarks.landmark_list)
        {
            double xdiff = lm.x_f-p.x;
            double ydiff = lm.y_f-p.y;

            //std::cout << "Landmark " << lm.x_f << "," << lm.y_f;

            if (xdiff*xdiff + ydiff*ydiff < maxDist)
            {
                //std::cout << " (OK)" << std::endl;
                predicted.push_back({lm.id_i, lm.x_f, lm.y_f});
            }
        }

        for (const auto& obs : observations)
        {
            // transform observed landmarks to map space
            double vx = p.x + (cos(p.theta)*obs.x - sin(p.theta)*obs.y);
            double vy = p.y + (sin(p.theta)*obs.x + cos(p.theta)*obs.y);

            //std::cout << obs.x << "," << obs.y << " -> " << vx << "," << vy << std::endl;
            transformed.push_back({obs.id, vx, vy});
        }

        //std::cout << "obs size: " << observations.size() << " pred size: " << predicted.size() << std::endl;
        //assert(observations.size() == predicted.size());

        // pair observations to predicted landmarks
        dataAssociation(predicted, transformed);

        p.weight = 1;

        // Update weights based on this
        for (size_t i = 0; i < transformed.size(); ++i)
        {
            const auto& obs = transformed[i];
            const auto& pred = predicted[i];
            const auto dx = obs.x - pred.x;
            const auto dy = obs.y - pred.y;
            const auto& ox = std_landmark[0];
            const auto& oy = std_landmark[1];

            /*
            std::cout << "Observed " << obs.x << "," << obs.y << " predicted " << pred.x << "," << pred.y << " *= " << ((1/(2*M_PI*ox*oy)) * exp(-(dx*dx/(2*ox*ox) + dy*dy/(2*oy*oy))))<< std::endl;
            std::cout << "2*M_PI*ox*oy " << 2*M_PI*ox*oy << " ox " << ox << " oy "<<  oy << " dx " << dx << " dy " << dy << std::endl;
            std::cout << "exp: " << exp(-(dx*dx/(2*ox*ox) + dy*dy/(2*oy*oy))) << std::endl;
            std::cout << "dxdx/2oxox " << dx*dx/(2*ox*ox) << " dy*dy/(2*oy*oy))) " << dy*dy/(2*oy*oy) << std::endl;

            */
            p.weight *= (1/(2*M_PI*ox*oy)) * exp(-(dx*dx/(2*ox*ox) + dy*dy/(2*oy*oy)));
        }
        std::cout << "Final weight: " << p.weight << std::endl;
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

    /*
    const double maxw = **std::max_element(particles.begin(), particles.end(),
            [](const auto& a, const auto& b) { return a.weight < b.weight; });
    const double beta = 2 * maxw;


    for (p : particles)
    {
    }
    */
  
    // Vector of weights of all particles
    std::vector<double> weights; 
    std::transform(particles.begin(), particles.end(), std::back_inserter(weights),
            [](const Particle& p) { return p.weight; });

    std::discrete_distribution<> disc(weights.begin(), weights.end());

    const auto old_particles(std::move(particles));

    for (size_t i = 0; i < old_particles.size(); ++i)
        std::cout << i << ". p: " << weights[i] << std::endl;

    for (size_t i = 0; i < old_particles.size(); ++i)
    {
        auto d = disc(gen);
        std::cout << i << ". -> " << d << std::endl;
        particles.push_back(old_particles[d]);
    }
}

  /*
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

  */
/*
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
*/
