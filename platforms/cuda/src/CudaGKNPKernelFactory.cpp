/* -------------------------------------------------------------------------- *
 *                            OpenMM-GKNP                                   *
 * -------------------------------------------------------------------------- */

#include <exception>

#include "CudaGKNPKernelFactory.h"
#include "CudaGKNPKernels.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace GKNPPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("Cuda");
        CudaGKNPKernelFactory* factory = new CudaGKNPKernelFactory();
        platform.registerKernelFactory(CalcGKNPForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerGKNPCudaKernelFactories() {
    try {
        Platform::getPlatformByName("Cuda");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaGKNPKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcGKNPForceKernel::Name())
        return new CudaCalcGKNPForceKernel(name, platform, cu, context.getSystem());
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
