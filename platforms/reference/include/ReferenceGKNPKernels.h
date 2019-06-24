#ifndef REFERENCE_GKNP_KERNELS_H_
#define REFERENCE_GKNP_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                           OpenMM-GKNP                                    *
 * -------------------------------------------------------------------------- */
#include "GKNPKernels.h"
#include "openmm/Platform.h"
#include <vector>
#include "gaussvol.h"

namespace GKNPPlugin {

/**
 * This kernel is invoked by GKNPForce to calculate the forces acting 
 * on the system and the energy of the system.
 */
class ReferenceCalcGKNPForceKernel : public CalcGKNPForceKernel {
public:
    ReferenceCalcGKNPForceKernel(std::string name, const OpenMM::Platform& platform) : CalcGKNPForceKernel(name, platform) {
    gvol = 0;
    }
  ~ReferenceCalcGKNPForceKernel(){
    if(gvol) delete gvol;
    positions.clear();
    ishydrogen.clear();
    radii_vdw.clear();
    radii_large.clear();
    gammas.clear();
    vdw_alpha.clear();
    charge.clear();
    free_volume.clear();
    self_volume.clear();
    vol_force.clear();
    vol_dv.clear();
    volume_scaling_factor.clear();
  }


    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the GKNPForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const GKNPForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);

    
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GKNPForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const GKNPForce& force);
 
private:
    GaussVol *gvol; // gaussvol instance
    //inputs
    int numParticles;
    std::vector<RealVec> positions;
    std::vector<int> ishydrogen;
    std::vector<RealOpenMM> radii_vdw;
    std::vector<RealOpenMM> radii_large;
    std::vector<RealOpenMM> gammas;
    double common_gamma;
    std::vector<RealOpenMM> vdw_alpha;
    std::vector<RealOpenMM> charge;
    //outputs
    std::vector<RealOpenMM> free_volume, self_volume;
    std::vector<RealOpenMM> free_volume_vdw, self_volume_vdw;
    std::vector<RealOpenMM> free_volume_large, self_volume_large;
    std::vector<RealVec> vol_force;
    std::vector<RealOpenMM> vol_dv;
    std::vector<RealOpenMM> volume_scaling_factor;
    double solvent_radius;
    double roffset;    
    double executeGVolSA(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    
};


 //class to record MS particles
 class MSParticle {
 public:
   double vol;
   double vol_large;
   double vol_vdw;
   double vol0;
   double ssp_large;
   double ssp_vdw;
   RealVec pos;
   int parent1;
   int parent2;
   RealVec gder;//used for volume derivatives
   RealVec hder;//used for positional derivatives
   double fms;
   double G0_vdw; //accumulator for derivatives
   double G0_large;
 };

 
} // namespace GKNPPlugin

#endif /*REFERENCE_GKNP_KERNELS_H_*/
