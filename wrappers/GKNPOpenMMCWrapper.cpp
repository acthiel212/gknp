
#include "OpenMM.h"
#include "GKNPOpenMMCWrapper.h"
#include <cstring>
#include <vector>

using namespace OpenMM;
using namespace std;

extern "C" {

/* OpenMM::GKNPForce */
OPENMM_EXPORT_GKNP OpenMM_GKNPForce* OpenMM_GKNPForce_create() {
    return reinterpret_cast<OpenMM_GKNPForce*>(new OpenMM::GKNPForce());
}
OPENMM_EXPORT_GKNP void OpenMM_GKNPForce_destroy(OpenMM_GKNPForce* target) {
    delete reinterpret_cast<OpenMM::GKNPForce*>(target);
}

}

