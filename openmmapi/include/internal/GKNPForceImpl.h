#ifndef OPENMM_GKNPFORCEIMPL_H_
#define OPENMM_GKNPFORCEIMPL_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM-GKNP                              *
 * -------------------------------------------------------------------------- */

#include "GKNPForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <string>

namespace GKNPPlugin {

class System;

/**
 * This is the internal implementation of GKNPForce.
 */

class OPENMM_EXPORT_GKNP GKNPForceImpl : public OpenMM::ForceImpl {
public:
    GKNPForceImpl(const GKNPForce& owner);
    ~GKNPForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const GKNPForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context,  bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(OpenMM::ContextImpl& context);
private:
    const GKNPForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace GKNPPlugin

#endif /*OPENMM_GKNPFORCEIMPL_H_*/
