#ifndef GKNP_KERNELS_H_
#define GKNP_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMM-GKNP                                 *
 * -------------------------------------------------------------------------- */

#include "GKNPForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace GKNPPlugin {

/**
 * This kernel is invoked by GKNPForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcGKNPForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcGKNPForce";
    }
    CalcGKNPForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the GKNPForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const GKNPForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GKNPForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const GKNPForce& force) = 0;
};

} // namespace GKNPPlugin

#endif /*GKNP_KERNELS_H_*/
