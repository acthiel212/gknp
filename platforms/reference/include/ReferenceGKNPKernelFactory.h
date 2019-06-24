#ifndef OPENMM_REFERENCEGKNPKERNELFACTORY_H_
#define OPENMM_REFERENCEGKNPKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM-GKNP                              *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the reference implementation of the 
 * GKNP plugin.
 */

class ReferenceGKNPKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_REFERENCEGKNPKERNELFACTORY_H_*/
