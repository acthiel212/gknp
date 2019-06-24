/* -------------------------------------------------------------------------- *
 *                             OpenMM-GKNP                                  *
 * -------------------------------------------------------------------------- */

#include <iostream>
#include "GKNPForce.h"
#include "internal/GKNPForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace GKNPPlugin;
using namespace OpenMM;
using namespace std;

GKNPForce::GKNPForce() : nonbondedMethod(NoCutoff), cutoffDistance(1.0), solvent_radius(SOLVENT_RADIUS) {
}

int GKNPForce::addParticle(double radius, double gamma, double vdw_alpha, double charge, bool ishydrogen){
  ParticleInfo particle(radius, gamma, vdw_alpha, charge, ishydrogen);
  particles.push_back(particle);
  return particles.size()-1;
}

void GKNPForce::setParticleParameters(int index, double radius, double gamma, double vdw_alpha, double charge, bool ishydrogen){
  ASSERT_VALID_INDEX(index, particles);
  particles[index].radius = radius;
  particles[index].radius;
  particles[index].gamma = gamma;
  particles[index].vdw_alpha = vdw_alpha;
  particles[index].charge = charge;
  particles[index].ishydrogen = ishydrogen;
}

GKNPForce::NonbondedMethod GKNPForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void GKNPForce::setNonbondedMethod(NonbondedMethod method) {
    nonbondedMethod = method;
}

double GKNPForce::getCutoffDistance() const {
    return cutoffDistance;
}

void GKNPForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

void GKNPForce::getParticleParameters(int index,  double& radius, double& gamma, double &vdw_alpha, double &charge,
				      bool& ishydrogen) const { 

    ASSERT_VALID_INDEX(index, particles);
    radius = particles[index].radius;
    gamma = particles[index].gamma;
    vdw_alpha = particles[index].vdw_alpha;
    charge = particles[index].charge;
    ishydrogen = particles[index].ishydrogen;
}

ForceImpl* GKNPForce::createImpl() const {
    return new GKNPForceImpl(*this);
}

void GKNPForce::updateParametersInContext(Context& context) {
    dynamic_cast<GKNPForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
