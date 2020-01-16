#ifndef OPENMM_GKNPFORCE_H_
#define OPENMM_GKNPFORCE_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM-GKNP                                        *
 * -------------------------------------------------------------------------- */

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <vector>
#include "internal/windowsExportGKNP.h"

//use nm and kj
#define ANG (0.1f)
#define ANG2 (.01f)
#define ANG3 (0.001f)

//volume cutoffs for MS spheres
//those for atomic overlaps are defined in gaussvol.h
//#define VOLMINMSA (0.25f*ANG3)
//#define VOLMINMSB (0.5f*ANG3)
#define VOLMINMSA (0.25f*ANG3)
#define VOLMINMSB (1.00f*ANG3)

//radius offset for surf energy calc.
#define GKNP_RADIUS_INCREMENT (0.000005f)

#define GKNP2_RADIUS_INCREMENT (0.1f*ANG)

//radius of a water molecule for GKNP2
#define SOLVENT_RADIUS (1.0*ANG)

//solvent radius for vdw energy
#define GKNP_HB_RADIUS (1.4*ANG)

namespace GKNPPlugin {

/* This class implements the GKNP implicit solvation model */

class OPENMM_EXPORT_GKNP GKNPForce : public OpenMM::Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling long range nonbonded forces.
     */
    enum NonbondedMethod {
        /**
         * No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,
        /**
         * Interactions beyond the cutoff distance are ignored.
         */
        CutoffNonPeriodic = 1,
        /**
         * Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy of
         * each other particle.  Interactions beyond the cutoff distance are ignored.
         */
        CutoffPeriodic = 2,
    };
    /**
     * Create an GKNPForce.
     */
    GKNPForce();
    /**
     * Add a particle to GKNP
     *
     * This should be called once for each particle in the System. When it is called for the i'th time, 
     * it specifies the parameters for the i'th particle.
     *
     * @param radius      the van der Waals radius of the particle, measured in nm
     * @param gamma       the surface tension parameter, measured in kJ/mol/nm^2
     * @param vdw_alpha   van der Waals solute-solvent interaction energy parameter
     * @param charge      electrostatic charge
     * @param ishydrogen  if true, this particle is a hydrogen atom (does not contribute to volume)
     * @return the index of the particle that was added
     */
    int addParticle(double radius, double gamma, double vdw_alpha, double charge, bool ishydrogen);

    /**
     * Modify the parameters of a particle to GKNP
     *
     * @param index       the index of the particle
     * @param radius      the van der Waals radius of the particle, measured in nm
     * @param gamma       the surface tension parameter, measured in kJ/mol/nm^2
     * @param vdw_alpha   van der Waals solute-solvent interaction energy parameter
     * @param charge      electrostatic charge
     * @param ishydrogen  if true, this particle is a hydrogen atom (does not contribute to volume)
     * @return the index of the particle that was added
     */
    void setParticleParameters(int index, double radius, double gamma, double vdw_alpha, double charge, bool ishydrogen);
    
    /**
     * Get the GKNP parameters for a particle
     * 
     * @param index       the index of the particle
     * @param radius      the van der Waals radius of the particle, measured in nm
     * @param gamma       the surface tension parameter, measured in kJ/mol/nm^2
     * @param vdw_alpha   van der Waals solute-solvent interaction energy parameter
     * @param charge      electric charge
     * @param ishydrogen  if true, this particle is a hydrogen atom
     */
    void getParticleParameters(int index, double& radius, double& gamma,  double& vdw_alpha, double &charge,
			       bool& ishydrogen) const;
    /**
     * Get the number of particles defined for GKNP
     */
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Get the method used for handling long range nonbonded interactions.
     */
    NonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling long range nonbonded interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);
    /**
     * Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);

    double getSolventRadius() const {
      return solvent_radius;
    }
    
    void updateParametersInContext(OpenMM::Context& context);


protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class ParticleInfo;
    std::vector<ParticleInfo> particles;
    NonbondedMethod nonbondedMethod;
    double cutoffDistance;
    double solvent_radius;
};

/**
 * This is an internal class used to record information about particles.
 * @private
 */
class GKNPForce::ParticleInfo {
 public:
  bool ishydrogen;
  double radius, gamma, vdw_alpha, charge;
  ParticleInfo() {
    ishydrogen = false;
    radius = 0.15;
    gamma = 0.0;
    charge = 0.0;
    vdw_alpha = 0.0;
  }
 ParticleInfo(double radius, double gamma, double vdw_alpha, double charge, bool ishydrogen) :
  radius(radius), gamma(gamma), vdw_alpha(vdw_alpha), charge(charge), ishydrogen(ishydrogen) {  }
 };
 
} // namespace GKNPPlugin

#endif /*OPENMM_GKNPFORCE_H_*/
