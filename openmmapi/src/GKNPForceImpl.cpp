/* -------------------------------------------------------------------------- *
 *                            OpenMM-GKNP                                   *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "internal/GKNPForceImpl.h"
#include "GKNPKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <map>
#include <set>
#include <sstream>

using namespace GKNPPlugin;
using namespace OpenMM;
using namespace std;

GKNPForceImpl::GKNPForceImpl(const GKNPForce& owner) : owner(owner) {
}

GKNPForceImpl::~GKNPForceImpl() {
}

void GKNPForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcGKNPForceKernel::Name(), context);
    kernel.getAs<CalcGKNPForceKernel>().initialize(context.getSystem(), owner);
}

double GKNPForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
  if ((groups&(1<<owner.getForceGroup())) != 0)
    return kernel.getAs<CalcGKNPForceKernel>().execute(context, includeForces, includeEnergy);
  return 0.0;
}

std::vector<std::string> GKNPForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcGKNPForceKernel::Name());
    return names;
}

void GKNPForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcGKNPForceKernel>().copyParametersToContext(context, owner);
}
