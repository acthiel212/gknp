
#ifndef GKNP_OPENMM_CWRAPPER_H_
#define GKNP_OPENMM_CWRAPPER_H_

#ifndef OPENMM_EXPORT_GKNP
#define OPENMM_EXPORT_GKNP
#endif
/* Global Constants */


/* Type Declarations */
typedef struct OpenMM_GKNPForce_struct OpenMM_GKNPForce;

#if defined(__cplusplus)
extern "C" {
#endif

/* GKNPForce */
extern OPENMM_EXPORT_GKNP OpenMM_GKNPForce* OpenMM_GKNPForce_create();
extern OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_destroy(OpenMM_GKNPForce* target);

#if defined(__cplusplus)
}
#endif

#endif /*GKNP_OPENMM_CWRAPPER_H_*/

