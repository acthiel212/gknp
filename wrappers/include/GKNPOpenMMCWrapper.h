
#ifndef GKNP_OPENMM_CWRAPPER_H_
#define GKNP_OPENMM_CWRAPPER_H_

#ifndef OPENMM_EXPORT_GKNP
#define OPENMM_EXPORT_GKNP
#endif
/* Global Constants */


/* Type Declarations */
typedef struct OpenMM_GKNPForce_struct OpenMM_GKNPForce;

/**
  * This is an enumeration of the different methods that may be used for handling long range nonbonded forces.
  */
typedef enum {
  OpenMM_GKNPForce_NoCutoff = 0, 
  OpenMM_GKNPForce_CutoffNonPeriodic = 1, 
  OpenMM_GKNPForce_CutoffPeriodic = 2 
} OpenMM_GKNPForce_NonbondedMethod;

#if defined(__cplusplus)
extern "C" {
#endif

/* GKNPForce */
extern OPENMM_EXPORT_GKNP OpenMM_GKNPForce* OpenMM_GKNPForce_create();
extern OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_destroy(OpenMM_GKNPForce* target);
extern OPENMM_EXPORT_GKNP int OpenMM_GKNPForce_addParticle(OpenMM_GKNPForce* target, 
                              double radius, double gamma, double vdw_alpha, double charge, OpenMM_Boolean ishydrogen);
extern OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_setParticleParameters(OpenMM_GKNPForce* target,
                               int index, double radius, double gamma, double vdw_alpha, double charge, OpenMM_Boolean ishydrogen);
extern OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_getParticleParameters(const OpenMM_GKNPForce* target,
                               int index, double* radius, double* gamma, double* vdw_alpha, double* charge, OpenMM_Boolean* ishydrogen);
extern OPENMM_EXPORT_GKNP int OpenMM_GKNPForce_getNumParticles(const OpenMM_GKNPForce* target);
extern OPENMM_EXPORT_GKNP OpenMM_GKNPForce_NonbondedMethod OpenMM_GKNPForce_getNonbondedMethod(const OpenMM_GKNPForce* target);
extern OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_setNonbondedMethod(OpenMM_GKNPForce* target, OpenMM_GKNPForce_NonbondedMethod method);
extern OPENMM_EXPORT_GKNP double OpenMM_GKNPForce_getCutoffDistance(const OpenMM_GKNPForce* target);
extern OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_setCutoffDistance(OpenMM_GKNPForce* target, double distance);
extern OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_updateParametersInContext(OpenMM_GKNPForce* target, OpenMM_Context* context);

#if defined(__cplusplus)
}
#endif

#endif /*GKNP_OPENMM_CWRAPPER_H_*/

