/* -------------------------------------------------------------------------- *
 *                              OpenMM-GKNP                                 *
 * -------------------------------------------------------------------------- */

#include "ReferenceGKNPKernelFactory.h"
#include "ReferenceGKNPKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace GKNPPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceGKNPKernelFactory* factory = new ReferenceGKNPKernelFactory();
            platform.registerKernelFactory(CalcGKNPForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerGKNPReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceGKNPKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcGKNPForceKernel::Name())
        return new ReferenceCalcGKNPForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
