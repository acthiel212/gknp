
#include "OpenMM.h"
#include "GKNPForce.h"
#include "OpenMMCWrapper.h"
#include "GKNPOpenMMCWrapper.h"
#include <cstring>
#include <vector>

using namespace GKNPPlugin;
using namespace OpenMM;
using namespace std;

extern "C" {

/* GKNPPlugin::GKNPForce */

OPENMM_EXPORT_GKNP OpenMM_GKNPForce* OpenMM_GKNPForce_create() {
    return reinterpret_cast<OpenMM_GKNPForce*>(new GKNPPlugin::GKNPForce());
}
OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_destroy(OpenMM_GKNPForce* target) {
    delete reinterpret_cast<GKNPPlugin::GKNPForce*>(target);
}
OPENMM_EXPORT_GKNP int OpenMM_GKNPForce_addParticle(OpenMM_GKNPForce* target, 
                       double radius, double gamma, double vdw_alpha, double charge, OpenMM_Boolean isHydrogen) {
    return reinterpret_cast<GKNPPlugin::GKNPForce*>(target)->addParticle(radius, gamma, vdw_alpha, charge, isHydrogen);
}
OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_setParticleParameters(OpenMM_GKNPForce* target,
                        int index, double radius, double gamma, double vdw_alpha, double charge, OpenMM_Boolean isHydrogen){
    reinterpret_cast<GKNPPlugin::GKNPForce*>(target)->setParticleParameters(index, radius, gamma, vdw_alpha, charge, isHydrogen);
}
OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_getParticleParameters(const OpenMM_GKNPForce* target,
                        int index, double* radius, double* gamma,  double* vdw_alpha, double* charge, OpenMM_Boolean* isHydrogen) {
                        reinterpret_cast<const GKNPPlugin::GKNPForce*>(target)->getParticleParameters(index, 
                        *reinterpret_cast<double*>(radius), *reinterpret_cast<double*>(gamma), 
                        *reinterpret_cast<double*>(vdw_alpha), *reinterpret_cast<double*>(charge),
                        *reinterpret_cast<bool*>(isHydrogen));
}
OPENMM_EXPORT_GKNP int OpenMM_GKNPForce_getNumParticles(const OpenMM_GKNPForce* target){
    return reinterpret_cast<const GKNPPlugin::GKNPForce*>(target)->getNumParticles(); 
}
OPENMM_EXPORT_GKNP OpenMM_GKNPForce_NonbondedMethod OpenMM_GKNPForce_getNonbondedMethod(const OpenMM_GKNPForce* target){
    GKNPPlugin::GKNPForce::NonbondedMethod result = reinterpret_cast<const GKNPPlugin::GKNPForce*>(target)->getNonbondedMethod(); 
    return static_cast<OpenMM_GKNPForce_NonbondedMethod>(result);
}
OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_setNonbondedMethod(OpenMM_GKNPForce* target, OpenMM_GKNPForce_NonbondedMethod method){
    reinterpret_cast<GKNPPlugin::GKNPForce*>(target)->setNonbondedMethod(static_cast<GKNPPlugin::GKNPForce::NonbondedMethod>(method));
}
OPENMM_EXPORT_GKNP double OpenMM_GKNPForce_getCutoffDistance(const OpenMM_GKNPForce* target){
    return reinterpret_cast<const GKNPPlugin::GKNPForce*>(target)->getCutoffDistance();
}
OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_setCutoffDistance(OpenMM_GKNPForce* target, double distance){
    reinterpret_cast<GKNPPlugin::GKNPForce*>(target)->setCutoffDistance(distance);
}
OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_updateParametersInContext(OpenMM_GKNPForce* target, OpenMM_Context* context){
    reinterpret_cast<GKNPPlugin::GKNPForce*>(target)->updateParametersInContext(*reinterpret_cast<Context*>(context));
}

}

