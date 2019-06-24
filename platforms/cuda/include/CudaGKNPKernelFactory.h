#ifndef OPENMM_CUDAGKNPKERNELFACTORY_H_
#define OPENMM_CUDAGKNPKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMM-GKNP                                 *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelFactory.h"

namespace GKNPPlugin {

/**
 * This KernelFactory creates kernels for the Cuda implementation of the GKNP plugin.
 */

class CudaGKNPKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace GKNPPlugin

#endif /*OPENMM_CUDAGKNPKERNELFACTORY_H_*/
