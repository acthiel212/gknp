/* -------------------------------------------------------------------------- *
 *                               OpenMM-GKNP                                *
 * -------------------------------------------------------------------------- */

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "ReferenceGKNPKernels.h"
#include "GKNPForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/SplineFitter.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include "gaussvol.h"


using namespace GKNPPlugin;
using namespace OpenMM;
using namespace std;


static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}


// Initializes GKNP library
void ReferenceCalcGKNPForceKernel::initialize(const System& system, const GKNPForce& force) {
   

    numParticles = force.getNumParticles();
    
    //input lists
    positions.resize(numParticles);
    radii_large.resize(numParticles);//van der Waals radii + offset (large radii)
    radii_vdw.resize(numParticles);//van der Waals radii (small radii)
    gammas.resize(numParticles);
    vdw_alpha.resize(numParticles);
    charge.resize(numParticles);
    ishydrogen.resize(numParticles);

    //output lists
    free_volume.resize(numParticles);
    self_volume.resize(numParticles);
    free_volume_vdw.resize(numParticles);
    self_volume_vdw.resize(numParticles);
    free_volume_large.resize(numParticles);
    self_volume_large.resize(numParticles);
    vol_force.resize(numParticles);
    vol_dv.resize(numParticles);

    vector<double> vdwrad(numParticles);
    roffset = GKNP_RADIUS_INCREMENT;
    common_gamma = -1;
    for (int i = 0; i < numParticles; i++){
      double r, g, alpha, q;
      bool h;
      force.getParticleParameters(i, r, g, alpha, q, h);
      radii_large[i] = r + roffset;
      radii_vdw[i] = r;
      vdwrad[i] = r; //double version for lookup table setup
      gammas[i] = g;
      if(h) gammas[i] = 0.0;
      vdw_alpha[i] = alpha;
      charge[i] = q;
      ishydrogen[i] = h ? 1 : 0;
      //make sure that all gamma's are the same
      if(common_gamma < 0 && !h){
	common_gamma = g; //first occurrence of a non-zero gamma
      }else{
	if(!h && pow(common_gamma - g,2) > FLT_MIN){
	  throw OpenMMException("initialize(): GKNP does not support multiple gamma values.");
	}
      }

    }

    //create and saves GaussVol instance
    //radii, volumes, etc. will be set in execute()
    gvol = new GaussVol(numParticles, ishydrogen);

    //volume scaling factors and born radii
    volume_scaling_factor.resize(numParticles);
    solvent_radius = force.getSolventRadius();
}

double ReferenceCalcGKNPForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
  double energy = 0.0;
  energy = executeGVolSA(context, includeForces, includeEnergy);
  return energy;
}


double ReferenceCalcGKNPForceKernel::executeGVolSA(ContextImpl& context, bool includeForces, bool includeEnergy) {

  //sequence: volume1->volume2

  
  //weights
    RealOpenMM w_evol = 1.0;
    
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    RealOpenMM energy = 0.0;
    int verbose_level = 0;

    vector<RealOpenMM> nu(numParticles);


    if(verbose_level > 0) cout << "Executing GVolSA" << endl;
    
    if(verbose_level > 0){
      cout << "-----------------------------------------------" << endl;
    } 

    
    // volume energy function 1 (large radii)
    RealOpenMM volume1, vol_energy1;

    gvol->setRadii(radii_large);

    vector<RealOpenMM> volumes_large(numParticles);
    for(int i = 0; i < numParticles; i++){
      volumes_large[i] = ishydrogen[i]>0 ? 0.0 : 4.*M_PI*pow(radii_large[i],3)/3.;
    }
    gvol->setVolumes(volumes_large);
    
    for(int i = 0; i < numParticles; i++){
      nu[i] = gammas[i]/roffset;
    }
    gvol->setGammas(nu);
    
    gvol->compute_tree(pos);
    gvol->compute_volume(pos, volume1, vol_energy1, vol_force, vol_dv, free_volume, self_volume);
    if(verbose_level > 2) gvol->print_tree();
      
    //returns energy and gradients from volume energy function
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol;
      if(verbose_level > 1) printf("self_volume: %6.6f atom: %d\n", self_volume[i], i);
      if(verbose_level > 1) printf("vol_energies[%d]: %6.6f\n", i, self_volume[i]*nu[i]);
    }
    energy += vol_energy1 * w_evol;

#ifdef NOTNOW
    //test of vol_dv
    vector<RealOpenMM> vol_dv2(numParticles);
    double energy_save = vol_energy1;
    for(int test_atom = 0; test_atom < numParticles; test_atom++){
      if(ishydrogen[test_atom]>0) continue;
      double deltav = -0.001*volumes_large[test_atom];
      double save_vol = volumes_large[test_atom];
      volumes_large[test_atom] += deltav;
      gvol->setVolumes(volumes_large);
      gvol->compute_tree(pos);
      gvol->compute_volume(pos, volume1, vol_energy1, vol_force, vol_dv2, free_volume, self_volume);
      cout << "DVV " << test_atom << " " << vol_energy1 - energy_save << " " << deltav*vol_dv[test_atom] << endl;
      volumes_large[test_atom] = save_vol;
    }
#endif
    
    // volume energy function 2 (small radii)
    RealOpenMM vol_energy2, volume2;

    gvol->setRadii(radii_vdw);

    vector<RealOpenMM> volumes_vdw(numParticles);
    for(int i = 0; i < numParticles; i++){
      volumes_vdw[i] = ishydrogen[i]>0 ? 0.0 : 4.*M_PI*pow(radii_vdw[i],3)/3.;
    }
    gvol->setVolumes(volumes_vdw);
    
    for(int i = 0; i < numParticles; i++){
      nu[i] = -gammas[i]/roffset;
    }
    gvol->setGammas(nu);

    gvol->rescan_tree_volumes(pos);
    gvol->compute_volume(pos, volume2, vol_energy2, vol_force, vol_dv, free_volume, self_volume);
    if(verbose_level > 2) gvol->print_tree();

 #ifdef NOTNOW
    //test of vol_dv
    vector<RealOpenMM> vol_dv2(numParticles);
    double energy_save = vol_energy2;
    for(int test_atom = 0; test_atom < numParticles; test_atom++){
      if(ishydrogen[test_atom]>0) continue;
      double deltav = -0.001*volumes_vdw[test_atom];
      double save_vol = volumes_vdw[test_atom];
      volumes_vdw[test_atom] += deltav;
      gvol->setVolumes(volumes_vdw);
      gvol->compute_tree(pos);
      gvol->compute_volume(pos, volume2, vol_energy2, vol_force, vol_dv2, free_volume, self_volume);
      cout << "DVV " << test_atom << " " << vol_energy2 - energy_save << " " << deltav*vol_dv[test_atom] << endl;
      volumes_vdw[test_atom] = save_vol;
    }
#endif

    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol;
        if(verbose_level > 1) printf("self_volume: %6.6f atom: %d\n", self_volume[i], i);
        if(verbose_level > 1) printf("vol_energies[%d]: %6.6f\n", i, self_volume[i]*nu[i]);
    }

    energy += vol_energy2 * w_evol;
    if(verbose_level > 0){
      cout << "Total number of overlaps in tree: " << gvol->getTotalNumberOfOverlaps() << endl;
    }
    
    //returns energy
    return (double)energy;
}


void ReferenceCalcGKNPForceKernel::copyParametersToContext(ContextImpl& context, const GKNPForce& force) {
  if (force.getNumParticles() != numParticles)
    throw OpenMMException("updateParametersInContext: The number of GKNP particles has changed");

  for (int i = 0; i < numParticles; i++){
    double r, g, alpha, q;
    bool h;
    force.getParticleParameters(i, r, g, alpha, q, h);
    if(pow(radii_vdw[i]-r,2) > 1.e-6){
      throw OpenMMException("updateParametersInContext: GKNP plugin does not support changing atomic radii.");
    }
    if(h && ishydrogen[i] == 0){
      throw OpenMMException("updateParametersInContext: GKNP plugin does not support changing heavy/hydrogen atoms.");
    }
    gammas[i] = g;
    if(h) gammas[i] = 0.0;
    vdw_alpha[i] = alpha;
    charge[i] = q;
  }
}
