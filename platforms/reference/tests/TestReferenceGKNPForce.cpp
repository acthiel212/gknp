//
// Created by Thiel, Andrew on 2019-06-07.
//

#include "OpenMM.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "GKNPForce.h"
#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace OpenMM;
using namespace GKNPPlugin;
using namespace std;

extern "C" OPENMM_EXPORT void registerGKNPReferenceKernelFactories();

static struct MyAtomInfo {
    const char* pdb;
    double      mass,vdwRadiusInAng, vdwVolume, gamma;
    bool        isHydrogen;
    double      charge;
    double      initPosInAng[3];
} atoms[] = {

        // pdb   mass vdwRad vdwVol gamma isHydrogen charge,  initPos
        {" C ", 12.00, 1.91, 28.28,  .16,  false, -0.18,  0.76506600,    0.00000200,   -0.00000100},
        {" C ", 12.00, 1.91, 28.28,  .16,  false, -0.18, -0.76506500,   -0.00000200,    0.00000100},
        {" H ",  1.00, 1.48, 12.77,  0,     true,   0.06, -1.16573300,    0.67232500,    0.77710400},
        {" H ",  1.00, 1.48, 12.77,  0,     true,   0.06, -1.16574800,    0.33683200,   -0.97079400},
        {" H ",  1.00, 1.48, 12.77,  0,     true,   0.06, -1.16572400,   -1.00915800,    0.19369400},
        {" H ",  1.00, 1.48, 12.77,  0,     true,   0.06,  1.16571800,    1.00915600,   -0.19370700},
        {" H ",  1.00, 1.48, 12.77,  0,     true,   0.06,  1.16574000,   -0.33682500,    0.97080200},
        {" H ",  1.00, 1.48, 12.77,  0,     true,   0.06,  1.16573600,   -0.67233000,   -0.77709700},
        {""} // end of list
};

void testEnergy() {

    System system;
    NonbondedForce *nb = new NonbondedForce();
    GKNPForce* force = new GKNPForce();
    force->setNonbondedMethod(GKNPForce::NoCutoff);//NoCutoff also accepted
    force->setCutoffDistance(1.0);
    system.addForce(nb);
    system.addForce(force);

    int numParticles = 8;
    vector<Vec3> positions;

    //Constants for unit/energy conversions
    double solventPressure = 0.11337;
    double volumeOffsetVdwToSEV = 27.939;
    double surfaceAreaOffsetVdwToSASA = 46.111;
    double surfaceTension = 0.16;
    //double rminToSigma = 1.0 / pow(2.0, 1.0 / 6.0);
    double ang2nm = 0.1;
    double kcalmol2kjmol = 4.184;
    double sigmaw = 3.15365*ang2nm; /* LJ sigma of TIP4P water oxygen */
    double epsilonw = 0.155*kcalmol2kjmol;        /* LJ epsilon of TIP4P water oxygen */
    double rho = 0.033428/pow(ang2nm,3);   /* water number density */
    double epsilon_LJ = 0.155*kcalmol2kjmol;
    double sigma_LJ;

    for(int i=0;i<numParticles;i++){

        system.addParticle(atoms[i].mass);
        positions.push_back(Vec3(atoms[i].initPosInAng[0], atoms[i].initPosInAng[1], atoms[i].initPosInAng[2])*ang2nm);
        atoms[i].vdwRadiusInAng *= ang2nm;
        sigma_LJ = 2.*atoms[i].vdwRadiusInAng;
        //atoms[i].vdwRadiusInAng *= rminToSigma;
        atoms[i].gamma *= kcalmol2kjmol/(ang2nm*ang2nm);
        double sij = sqrt(sigmaw*sigma_LJ);
        double eij = sqrt(epsilonw*epsilon_LJ);
        double alpha = - 16.0 * M_PI * rho * eij * pow(sij,6) / 3.0;
        nb->addParticle(0.0,0.0,0.0);
        force->addParticle(atoms[i].vdwRadiusInAng, atoms[i].gamma, alpha, atoms[i].charge, atoms[i].isHydrogen);
        cout << "Atom: " << i << " Radius: " << atoms[i].vdwRadiusInAng << " gamma: " << atoms[i].gamma << " alpha: " << alpha << " Charge: " << atoms[i].charge << " Hydrogen?: " << atoms[i].isHydrogen << endl;
        force->getParticleParameters(i, atoms[i].vdwRadiusInAng, atoms[i].gamma, alpha, atoms[i].charge, atoms[i].isHydrogen);
    }

    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("Reference");

    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces | State::Positions);

    double energy1 = 0;
    energy1 = state.getPotentialEnergy();
    double surfaceArea = (energy1/kcalmol2kjmol)/surfaceTension + surfaceAreaOffsetVdwToSASA;
    double surfaceAreaEnergy=surfaceArea*surfaceTension;
    cout << endl;
    cout << std::setw(25) << std::left << "Surface Area: " << std::fixed << surfaceArea << " (Ang^2)" << endl;
    cout << std::setw(25) << std::left << "Surface Area Energy: " << surfaceAreaEnergy << " (kcal/mol)" << endl << endl;

    cout << "Forces: " << endl;
    for(int i = 0; i < numParticles; i++){
        cout << "FW: " << i << " " << state.getForces()[i][0] << " " << state.getForces()[i][1] << " "  << state.getForces()[i][2] << " "<< endl;
    }

}

int main(){
    try {
        registerGKNPReferenceKernelFactories();
        testEnergy();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
