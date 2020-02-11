/* -------------------------------------------------------------------------- *
 *                                   OpenMM-GKNP                            *
 * -------------------------------------------------------------------------- */

#include "CudaGKNPKernels.h"
#include "CudaGKNPKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaNonbondedUtilities.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "cuda.h"
#include <cmath>
#include <cfloat>


#include <fstream>
#include <iomanip>
#include <algorithm>

#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/reference/RealVec.h"
#include "gaussvol.h"

//conversion factors
//#define ANG (0.1f)
//#define ANG3 (0.001f)

//volume cutoffs in switching function
//#define MIN_GVOL (FLT_MIN)
#define VOLMIN0 (0.009f*ANG3)
#define VOLMINA (0.01f*ANG3)
#define VOLMINB (0.1f*ANG3)

#ifndef PI
#define PI (3.14159265359)
#endif

// conversion factors from spheres to Gaussians
//#define KFC (2.2269859253)

//using namespace GKNPPlugin;
using namespace OpenMM;
using namespace std;


class CudaGKNPForceInfo : public CudaForceInfo {
public:
    CudaGKNPForceInfo(const GKNPPlugin::GKNPForce &force) : force(force) {
    }

    int getNumParticleGroups() {
        return force.getNumParticles();//each particle is in a different group?
    }

    void getParticlesInGroup(int index, vector<int> &particles) {
        particles.push_back(index);
    }

    bool areGroupsIdentical(int group1, int group2) {
        return (group1 == group2);
    }

private:
    const GKNPPlugin::GKNPForce &force;
};

GKNPPlugin::CudaCalcGKNPForceKernel::~CudaCalcGKNPForceKernel() {
    if (gtree != NULL) delete gtree;
}


static int _nov_ = 0;


//version based on number of overlaps for each atom
void GKNPPlugin::CudaCalcGKNPForceKernel::CudaOverlapTree::init_tree_size(int num_atoms,
                                                              int padded_num_atoms,
                                                              int num_compute_units,
                                                              int pad_modulo,
                                                              vector<int> &noverlaps_current) {
    this->num_atoms = num_atoms;
    this->padded_num_atoms = padded_num_atoms;

    total_tree_size = 0;
    tree_size.clear();
    tree_pointer.clear();
    padded_tree_size.clear();
    atom_tree_pointer.clear();
    natoms_in_tree.clear();
    first_atom.clear();

    //The tree may be reinitialized multiple times due to too many overlaps.
    //Remember the largest number of overlaps per atom because if it went over the max before it
    //is likely to happen again
    if (!has_saved_noverlaps) {
        saved_noverlaps.resize(num_atoms);
        for (int i = 0; i < num_atoms; i++) saved_noverlaps[i] = 0;
        has_saved_noverlaps = true;
    }
    vector<int> noverlaps(num_atoms);
    for (int i = 0; i < num_atoms; i++) {
        noverlaps[i] = (saved_noverlaps[i] > noverlaps_current[i]) ? saved_noverlaps[i] : noverlaps_current[i] + 1;
        //(the +1 above counts the 1-body overlap)
    }
    for (int i = 0; i < num_atoms; i++) saved_noverlaps[i] = noverlaps[i];

    //assigns atoms to compute units (tree sections) in such a way that each compute unit gets
    //approximately equal number of overlaps
    num_sections = num_compute_units;
    vector<int> noverlaps_sum(num_atoms + 1);//prefix sum of number of overlaps per atom
    noverlaps_sum[0] = 0;
    for (int i = 1; i <= num_atoms; i++) {
        noverlaps_sum[i] = noverlaps[i - 1] + noverlaps_sum[i - 1];
    }
    int n_overlaps_total = noverlaps_sum[num_atoms];

    //  for(int i=0;i < num_atoms;i++){
    //  cout << "nov " << i << " noverlaps " << noverlaps[i] << " " <<  noverlaps_sum[i] << endl;
    //}
    int max_n_overlaps = 0;
    for (int i = 0; i < num_atoms; i++) {
        if (noverlaps[i] > max_n_overlaps) max_n_overlaps = noverlaps[i];
    }

    int n_overlaps_per_section;
    if (num_sections > 1) {
        n_overlaps_per_section = n_overlaps_total / (num_sections - 1);
    } else {
        n_overlaps_per_section = n_overlaps_total;
    }
    if (max_n_overlaps > n_overlaps_per_section) n_overlaps_per_section = max_n_overlaps;

    //cout << "n_overlaps_per_section : " << n_overlaps_per_section << endl;


    //assigns atoms to compute units
    vector<int> compute_unit_of_atom(num_atoms);
    total_atoms_in_tree = 0;
    natoms_in_tree.resize(num_sections);
    for (int section = 0; section < num_sections; section++) {
        natoms_in_tree[section] = 0;
    }
    for (int i = 0; i < num_atoms; i++) {
        int section = noverlaps_sum[i] / n_overlaps_per_section;
        compute_unit_of_atom[i] = section;
        natoms_in_tree[section] += 1;
        total_atoms_in_tree += 1;
    }

    // computes sizes of tree sections
    vector<int> section_size(num_sections);
    for (int section = 0; section < num_sections; section++) {
        section_size[section] = 0;
    }
    for (int i = 0; i < num_atoms; i++) {
        int section = compute_unit_of_atom[i];
        section_size[section] += noverlaps[i];
    }
    //double sizes and pad for extra buffer
    for (int section = 0; section < num_sections; section++) {
        int tsize = section_size[section] < 1 ? 1 : section_size[section];
        tsize *= tree_size_boost;
        int npadsize = pad_modulo * ((tsize + pad_modulo - 1) / pad_modulo);
        section_size[section] = npadsize;
    }

    // set tree pointers
    tree_pointer.resize(num_sections);
    int offset = 0;
    for (int section = 0; section < num_sections; section++) {
        tree_pointer[section] = offset;
        offset += section_size[section];
    }

    // set atom pointer in tree
    tree_size.resize(num_sections);
    padded_tree_size.resize(num_sections);
    atom_tree_pointer.resize(padded_num_atoms);
    first_atom.resize(num_sections);
    int atom_offset = 0;
    for (int section = 0; section < num_sections; section++) {
        tree_size[section] = 0;
        padded_tree_size[section] = section_size[section];
        first_atom[section] = atom_offset;
        for (int i = 0; i < natoms_in_tree[section]; i++) {
            int iat = atom_offset + i;
            int slot = tree_pointer[section] + i;
            if (iat < total_atoms_in_tree) {
                atom_tree_pointer[iat] = slot;
            }
        }
        total_tree_size += section_size[section];
        atom_offset += natoms_in_tree[section];
    }

}

//simpler version with precomputed tree sizes
void GKNPPlugin::CudaCalcGKNPForceKernel::CudaOverlapTree::init_tree_size(int padded_num_atoms, int tree_section_size,
                                                              int num_compute_units, int pad_modulo) {
    this->num_atoms = padded_num_atoms;
    this->padded_num_atoms = padded_num_atoms;
    num_sections = num_compute_units;

    total_tree_size = 0;
    tree_size.clear();
    tree_pointer.clear();
    padded_tree_size.clear();
    atom_tree_pointer.clear();
    natoms_in_tree.clear();
    first_atom.clear();

    total_atoms_in_tree = 0;
    natoms_in_tree.resize(num_sections);
    for (int section = 0; section < num_sections; section++) {
        natoms_in_tree[section] = 0;
    }

    //tree sizes
    padded_tree_size.resize(num_sections);
    for (int section = 0; section < num_sections; section++) {
        int tsize = tree_section_size;
        int npadsize = pad_modulo * ((tsize + pad_modulo - 1) / pad_modulo);
        padded_tree_size[section] = npadsize;
    }

    // set tree pointers
    tree_pointer.resize(num_sections);
    int offset = 0;
    for (int section = 0; section < num_sections; section++) {
        tree_pointer[section] = offset;
        offset += padded_tree_size[section];
    }

    tree_size.resize(num_sections);
    for (int section = 0; section < num_sections; section++) tree_size[section] = 0;

    atom_tree_pointer.resize(padded_num_atoms);
    first_atom.resize(num_sections);
    int atom_offset = 0;
    for (int section = 0; section < num_sections; section++) {
        first_atom[section] = atom_offset;
        total_tree_size += padded_tree_size[section];
    }
}


void GKNPPlugin::CudaCalcGKNPForceKernel::CudaOverlapTree::resize_tree_buffers(OpenMM::CudaContext &cu, int ov_work_group_size) {
    if (ovAtomTreePointer) delete ovAtomTreePointer;
    ovAtomTreePointer = CudaArray::create<int>(cu, padded_num_atoms, "ovAtomTreePointer");
    if (ovAtomTreeSize) delete ovAtomTreeSize;
    ovAtomTreeSize = CudaArray::create<int>(cu, num_sections, "ovAtomTreeSize");
    if (NIterations) delete NIterations;
    NIterations = CudaArray::create<int>(cu, num_sections, "NIterations");
    if (ovAtomTreePaddedSize) delete ovAtomTreePaddedSize;
    ovAtomTreePaddedSize = CudaArray::create<int>(cu, num_sections, "ovAtomTreePaddedSize");
    if (ovNumAtomsInTree) delete ovNumAtomsInTree;
    ovNumAtomsInTree = CudaArray::create<int>(cu, num_sections, "ovNumAtomsInTree");
    if (ovTreePointer) delete ovTreePointer;
    ovTreePointer = CudaArray::create<int>(cu, num_sections, "ovTreePointer");
    if (ovAtomTreeLock) delete ovAtomTreeLock;
    ovAtomTreeLock = CudaArray::create<int>(cu, num_sections, "ovAtomTreeLock");
    if (ovFirstAtom) delete ovFirstAtom;
    ovFirstAtom = CudaArray::create<int>(cu, num_sections, "ovFirstAtom");
    if (ovLevel) delete ovLevel;
    ovLevel = CudaArray::create<int>(cu, total_tree_size, "ovLevel");
    if (ovG) delete ovG;
    ovG = CudaArray::create<float4>(cu, total_tree_size, "ovG"); //gaussian position + exponent
    if (ovVolume) delete ovVolume;
    ovVolume = CudaArray::create<float>(cu, total_tree_size, "ovVolume");
    if (ovVsp) delete ovVsp;
    ovVsp = CudaArray::create<float>(cu, total_tree_size, "ovVsp");
    if (ovVSfp) delete ovVSfp;
    ovVSfp = CudaArray::create<float>(cu, total_tree_size, "ovVSfp");
    if (ovSelfVolume) delete ovSelfVolume;
    ovSelfVolume = CudaArray::create<float>(cu, total_tree_size, "ovSelfVolume");
    if (ovVolEnergy) delete ovVolEnergy;
    ovVolEnergy = CudaArray::create<float>(cu, total_tree_size, "ovVolEnergy");
    if (ovGamma1i) delete ovGamma1i;
    ovGamma1i = CudaArray::create<float>(cu, total_tree_size, "ovGamma1i");
    if (ovDV1) delete ovDV1;
    ovDV1 = CudaArray::create<float4>(cu, total_tree_size, "ovDV1"); //dV12/dr1 + dV12/dV1 for each overlap
    if (ovDV2) delete ovDV2;
    ovDV2 = CudaArray::create<float4>(cu, total_tree_size, "ovDV2"); //volume gradient accumulator
    if (ovPF) delete ovPF;
    ovPF = CudaArray::create<float4>(cu, total_tree_size, "ovPF"); //(P) and (F) auxiliary variables
    if (ovLastAtom) delete ovLastAtom;
    ovLastAtom = CudaArray::create<int>(cu, total_tree_size, "ovLastAtom");
    if (ovRootIndex) delete ovRootIndex;
    ovRootIndex = CudaArray::create<int>(cu, total_tree_size, "ovRootIndex");
    if (ovChildrenStartIndex) delete ovChildrenStartIndex;
    ovChildrenStartIndex = CudaArray::create<int>(cu, total_tree_size, "ovChildrenStartIndex");
    if (ovChildrenCount) delete ovChildrenCount;
    ovChildrenCount = CudaArray::create<int>(cu, total_tree_size, "ovChildrenCount");
    if (ovChildrenCountTop) delete ovChildrenCountTop;
    ovChildrenCountTop = CudaArray::create<int>(cu, total_tree_size, "ovChildrenCountTop");
    if (ovChildrenCountBottom) delete ovChildrenCountBottom;
    ovChildrenCountBottom = CudaArray::create<int>(cu, total_tree_size, "ovChildrenCountBottom");
    if (ovProcessedFlag) delete ovProcessedFlag;
    ovProcessedFlag = CudaArray::create<int>(cu, total_tree_size, "ovProcessedFlag");
    if (ovOKtoProcessFlag) delete ovOKtoProcessFlag;
    ovOKtoProcessFlag = CudaArray::create<int>(cu, total_tree_size, "ovOKtoProcessFlag");
    if (ovChildrenReported) delete ovChildrenReported;
    ovChildrenReported = CudaArray::create<int>(cu, total_tree_size, "ovChildrenReported");


    // atomic reduction buffers, one for each tree section
    // used only if long int atomics are not available
    //   ovAtomBuffer holds volume energy derivatives (in xyz)
    if (ovAtomBuffer) delete ovAtomBuffer;
    ovAtomBuffer = CudaArray::create<float4>(cu, padded_num_atoms * num_sections, "ovAtomBuffer");

    //regular and "long" versions of selfVolume accumulation buffer (the latter updated using atomics)
    if (selfVolumeBuffer) delete selfVolumeBuffer;
    selfVolumeBuffer = CudaArray::create<float>(cu, padded_num_atoms * num_sections, "selfVolumeBuffer");
    if (selfVolumeBuffer_long) delete selfVolumeBuffer_long;
    selfVolumeBuffer_long = CudaArray::create<long>(cu, padded_num_atoms, "selfVolumeBuffer_long");

    //traditional and "long" versions of general accumulation buffers
    if (!AccumulationBuffer1_real) delete AccumulationBuffer1_real;
    AccumulationBuffer1_real = CudaArray::create<float>(cu, padded_num_atoms * num_sections,
                                                        "AccumulationBuffer1_real");
    if (!AccumulationBuffer1_long) delete AccumulationBuffer1_long;
    AccumulationBuffer1_long = CudaArray::create<long>(cu, padded_num_atoms, "AccumulationBuffer1_long");
    if (!AccumulationBuffer2_real) delete AccumulationBuffer2_real;
    AccumulationBuffer2_real = CudaArray::create<float>(cu, padded_num_atoms * num_sections,
                                                        "AccumulationBuffer2_real");
    if (!AccumulationBuffer2_long) delete AccumulationBuffer2_long;
    AccumulationBuffer2_long = CudaArray::create<long>(cu, padded_num_atoms, "AccumulationBuffer2_long");

    if (!gradBuffers_long) delete gradBuffers_long;
    gradBuffers_long = CudaArray::create<long>(cu, 4 * padded_num_atoms, "gradBuffers_long");

    //temp buffers to cache intermediate data in overlap tree construction (3-body and up)
    if (temp_buffer_size <= 0) {//first time
        int smax = 64; // this is n*(n-1)/2 where n is the max number of neighbors per overlap
        temp_buffer_size = ov_work_group_size * num_sections * smax;//first time
    }
    if (hasExceededTempBuffer) {//increase if needed
        temp_buffer_size = 2 * temp_buffer_size;
        hasExceededTempBuffer = false;
    }
    if (gvol_buffer_temp) delete gvol_buffer_temp;
    gvol_buffer_temp = CudaArray::create<float>(cu, temp_buffer_size, "gvol_buffer_temp");
    if (tree_pos_buffer_temp) delete tree_pos_buffer_temp;
    tree_pos_buffer_temp = CudaArray::create<unsigned int>(cu, temp_buffer_size, "tree_pos_buffer_temp");
    if (i_buffer_temp) delete i_buffer_temp;
    i_buffer_temp = CudaArray::create<int>(cu, temp_buffer_size, "i_buffer_temp");
    if (atomj_buffer_temp) delete atomj_buffer_temp;
    atomj_buffer_temp = CudaArray::create<int>(cu, temp_buffer_size, "atomj_buffer_temp");
}

int GKNPPlugin::CudaCalcGKNPForceKernel::CudaOverlapTree::copy_tree_to_device(void) {

    vector<int> nn(padded_num_atoms);
    vector<int> ns(num_sections);

    for (int i = 0; i < padded_num_atoms; i++) {
        nn[i] = (int) atom_tree_pointer[i];
    }
    ovAtomTreePointer->upload(nn);

    for (int i = 0; i < num_sections; i++) {
        ns[i] = (int) tree_pointer[i];
    }
    ovTreePointer->upload(ns);

    for (int i = 0; i < num_sections; i++) {
        ns[i] = (int) tree_size[i];
    }
    ovAtomTreeSize->upload(ns);

    for (int i = 0; i < num_sections; i++) {
        ns[i] = (int) padded_tree_size[i];
    }
    ovAtomTreePaddedSize->upload(ns);

    for (int i = 0; i < num_sections; i++) {
        ns[i] = (int) natoms_in_tree[i];
    }
    ovNumAtomsInTree->upload(ns);

    for (int i = 0; i < num_sections; i++) {
        ns[i] = (int) first_atom[i];
    }
    ovFirstAtom->upload(ns);

    return 1;
}

void GKNPPlugin::CudaCalcGKNPForceKernel::initialize(const System &system, const GKNPForce &force) {
    verbose_level = 5;

    roffset = GKNP_RADIUS_INCREMENT;



    //we do not support multiple contexts(?), is it the same as multiple devices?
    if (cu.getPlatformData().contexts.size() > 1)
        throw OpenMMException("GKNPForce does not support using multiple contexts");

    CudaNonbondedUtilities &nb = cu.getNonbondedUtilities();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));

    numParticles = cu.getNumAtoms();//force.getNumParticles();
    if (numParticles == 0)
        return;
    radiusParam1 = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(float), "radiusParam1");
    radiusParam2 = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(float), "radiusParam2");
    gammaParam1 = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(float), "gammaParam1");
    gammaParam2 = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(float), "gammaParam2");
    chargeParam = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(float), "chargeParam");
    alphaParam = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(float), "alphaParam");
    ishydrogenParam = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(int), "ishydrogenParam");

    testBuffer = new CudaArray(cu, cu.getPaddedNumAtoms(), sizeof(float), "testBuffer");

    //bool useLong = cu.getSupports64BitGlobalAtomics();

    // this the accumulation buffer for overlap atom-level data (self-volumes, etc.)
    // note that each thread gets a separate buffer of size Natoms (rather than each thread block as in the
    // non-bonded algorithm). This may limits the max number of atoms.

    //cu.addAutoclearBuffer(*ovAtomBuffer);

    radiusVector1.resize(cu.getPaddedNumAtoms());
    radiusVector2.resize(cu.getPaddedNumAtoms());
    gammaVector1.resize(cu.getPaddedNumAtoms());
    gammaVector2.resize(cu.getPaddedNumAtoms());
    chargeVector.resize(cu.getPaddedNumAtoms());
    alphaVector.resize(cu.getPaddedNumAtoms());
    ishydrogenVector.resize(cu.getPaddedNumAtoms());
    atom_ishydrogen.resize(cu.getPaddedNumAtoms());
    common_gamma = -1;
    for (int i = 0; i < numParticles; i++) {
        double radius, gamma, alpha, charge;
        bool ishydrogen;
        force.getParticleParameters(i, radius, gamma, alpha, charge, ishydrogen);
        radiusVector1[i] = (float) radius + roffset;
        radiusVector2[i] = (float) radius;

        atom_ishydrogen[i] = ishydrogen ? 1 : 0;
        ishydrogenVector[i] = ishydrogen ? 1 : 0;

        // for surface-area energy use gamma/radius_offset
        // gamma = 1 for self volume calculation.
        double g = ishydrogen ? 0 : gamma / roffset; //TODO: Possible cause of math discrepancy
        gammaVector1[i] = (float) g;
        gammaVector2[i] = (float) -g;
        alphaVector[i] = (float) alpha;
        chargeVector[i] = (float) charge;

        //make sure that all gamma's are the same
        if (common_gamma < 0 && !ishydrogen) {
            common_gamma = gamma; //first occurrence of a non-zero gamma
        } else {
            if (!ishydrogen && pow(common_gamma - gamma, 2) > 1.e-6f) {
                throw OpenMMException("initialize(): GKNP does not support multiple gamma values.");
            }
        }

    }
    radiusParam1->upload(radiusVector1);
    radiusParam2->upload(radiusVector2);
    gammaParam1->upload(gammaVector1);
    gammaParam2->upload(gammaVector2);
    alphaParam->upload(alphaVector);
    chargeParam->upload(chargeVector);
    ishydrogenParam->upload(ishydrogenVector);

    useCutoff = (force.getNonbondedMethod() != GKNPForce::NoCutoff);
    usePeriodic = (force.getNonbondedMethod() != GKNPForce::NoCutoff &&
                   force.getNonbondedMethod() != GKNPForce::CutoffNonPeriodic);
    useExclusions = false;
    cutoffDistance = force.getCutoffDistance();
    if (verbose_level > 1) {
        cout << "Cutoff distance: " << cutoffDistance << endl;
    }

    gtree = new CudaOverlapTree;//instance of atomic overlap tree


    gvol_force = &force;
    niterations = 0;
    hasInitializedKernels = false;
    hasCreatedKernels = false;
}

double GKNPPlugin::CudaCalcGKNPForceKernel::execute(ContextImpl &context, bool includeForces, bool includeEnergy) {
    double energy = 0.0;
    if (!hasCreatedKernels || !hasInitializedKernels) {
        executeInitKernels(context, includeForces, includeEnergy);
        hasInitializedKernels = true;
        hasCreatedKernels = true;
    }
    energy = executeGVolSA(context, includeForces, includeEnergy);
    return 0.0;
}

void GKNPPlugin::CudaCalcGKNPForceKernel::executeInitKernels(ContextImpl &context, bool includeForces, bool includeEnergy) {
    CudaNonbondedUtilities &nb = cu.getNonbondedUtilities();
    //bool useLong = cu.getSupports64BitGlobalAtomics();
    bool verbose = verbose_level > 0;

    maxTiles = (nb.getUseCutoff() ? nb.getInteractingTiles().getSize() : 0);

    //run CPU version once to estimate sizes
    {
        GaussVol *gvol;
        std::vector<RealVec> positions;
        std::vector<int> ishydrogen;
        std::vector<RealOpenMM> radii;
        std::vector<RealOpenMM> gammas;
        //outputs
        RealOpenMM volume, vol_energy;
        std::vector<RealOpenMM> free_volume, self_volume;
        std::vector<RealVec> vol_force;
        std::vector<RealOpenMM> vol_dv;
        int numParticles = cu.getNumAtoms();
        //input lists
        positions.resize(numParticles);
        radii.resize(numParticles);
        gammas.resize(numParticles);
        ishydrogen.resize(numParticles);
        //output lists
        free_volume.resize(numParticles);
        self_volume.resize(numParticles);
        vol_force.resize(numParticles);
        vol_dv.resize(numParticles);

        //double energy_density_param = 4.184 * 1000.0 / 27; //about 1 kcal/mol for each water volume
        //double energy_density_param = .08 * 4.184 /(0.1 * 0.1);
        for (int i = 0; i < numParticles; i++) {
            double r, g, alpha, q;
            bool h;
            gvol_force->getParticleParameters(i, r, g, alpha, q, h);
            radii[i] = r + roffset;
            gammas[i] = g / roffset; //energy_density_param;
            if (h) gammas[i] = 0.0;
            ishydrogen[i] = h ? 1 : 0;
        }
        gvol = new GaussVol(numParticles, ishydrogen);
        vector<float4> posq;
        cu.getPosq().download(posq);
        for (int i = 0; i < numParticles; i++) {
            positions[i] = RealVec((RealOpenMM) posq[i].x, (RealOpenMM) posq[i].y, (RealOpenMM) posq[i].z);
        }
        vector<RealOpenMM> volumes(numParticles);
        for (int i = 0; i < numParticles; i++) {
            volumes[i] = 4. * M_PI * pow(radii[i], 3) / 3.;
        }
        //CPU GaussVol really necessary?
            gvol->setRadii(radii);
            gvol->setVolumes(volumes);
            gvol->setGammas(gammas);
            gvol->compute_tree(positions);
            //gvol->compute_volume(positions, volume, vol_energy, vol_force, vol_dv, free_volume, self_volume);
            vector<int> noverlaps(cu.getPaddedNumAtoms());
            for (int i = 0; i < cu.getPaddedNumAtoms(); i++) noverlaps[i] = 0;
            gvol->getstat(noverlaps);
            gvol->print_tree();


            int nn = 0;
            for (int i = 0; i < noverlaps.size(); i++) {
                nn += noverlaps[i];
            }

        if (verbose_level > 0) cout << "Total number of overlaps in tree: " << nn << endl;

        //TODO: Query device properties in Cuda?
//      if(verbose_level > 0){
//	cout << "Device: " << cu.getDevice().getInfo<CL_DEVICE_NAME>()  << endl;
//	cout << "MaxSharedMem: " << cu.getDevice().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()  << endl;
//	cout << "CompUnits: " << cu.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()  << endl;
//	cout << "Max Work Group Size: " << cu.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()  << endl;
//	//cout << "Supports 64bit Atomics: " << useLong << endl;
//      }

        ov_work_group_size = nb.getForceThreadBlockSize();
        num_compute_units = nb.getNumForceThreadBlocks();

        //creates overlap tree
        int pad_modulo = ov_work_group_size;
        gtree->init_tree_size(cu.getNumAtoms(), cu.getPaddedNumAtoms(), num_compute_units, pad_modulo, noverlaps);
        //gtree->init_tree_size(cu.getNumAtoms(), cu.getPaddedNumAtoms(), num_compute_units, pad_modulo);
        //allocates or re-allocates tree buffers
        gtree->resize_tree_buffers(cu, ov_work_group_size);
        //copy overlap tree buffers to device
        gtree->copy_tree_to_device();

        if (verbose_level > 0) std::cout << "Tree size: " << gtree->total_tree_size << std::endl;

        if (verbose_level > 0) {
            for (int i = 0; i < gtree->num_sections; i++) {
                cout << "Tn: " << i << " " << gtree->tree_size[i] << " " << gtree->padded_tree_size[i] << " "
                     << gtree->natoms_in_tree[i] << " " << gtree->tree_pointer[i] << " " << gtree->first_atom[i]
                     << endl;
            }
            if (verbose_level > 4) {
                for (int i = 0; i < gtree->total_atoms_in_tree; i++) {
                    cout << "Atom: " << i << " Slot: " << gtree->atom_tree_pointer[i] << endl;
                }
            }
        }

        if (verbose_level > 0) {
            std::cout << "Num atoms: " << cu.getNumAtoms() << std::endl;
            std::cout << "Padded Num Atoms: " << cu.getPaddedNumAtoms() << std::endl;
            std::cout << "Num Atom Blocks: " << cu.getNumAtomBlocks() << std::endl;
            std::cout << "Num Tree Sections: " << gtree->num_sections << std::endl;
            std::cout << "Num Force Buffers: " << nb.getNumEnergyBuffers() << std::endl;
            std::cout << "Tile size: " << CudaContext::TileSize << std::endl;
            std::cout << "getNumForceThreadBlocks: " << nb.getNumForceThreadBlocks() << std::endl;
            std::cout << "getForceThreadBlockSize: " << nb.getForceThreadBlockSize() << std::endl;
            //std::cout << "numForceBuffers: " << nb.getNumForceBuffers() << std::endl;
            std::cout << "Num Tree Sections: " << gtree->num_sections << std::endl;
            std::cout << "Work Group Size: " << ov_work_group_size << std::endl;
            std::cout << "Tree Size: " << gtree->total_tree_size << std::endl;


            if (useCutoff) {
                vector<int> icount(1024);
                nb.getInteractionCount().download(icount);
                cout << "Using cutoff" << endl;
                cout << "Number of interacting tiles: " << icount[0] << endl;
            } else {
                cout << "Not using cutoff" << endl;
            }

        }


        delete gvol; //no longer needed

        //Sets up buffers
        //TODO: Panic Button?
        //sets up flag to detect when tree size is exceeded
        if (PanicButton) delete PanicButton;
        // pos 0 is general panic, pos 1 indicates execeeded temp buffer
        PanicButton = CudaArray::create<int>(cu, 2, "PanicButton");
        panic_button.resize(2);
        panic_button[0] = panic_button[1] = 0;    //init with zero
        PanicButton->upload(panic_button);

        // atom-level properties
        if (selfVolume) delete selfVolume;
        selfVolume = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "selfVolume");
        if (selfVolumeLargeR) delete selfVolumeLargeR;
        selfVolumeLargeR = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "selfVolumeLargeR");
        if (Semaphor) delete Semaphor;
        Semaphor = CudaArray::create<int>(cu, cu.getPaddedNumAtoms(), "Semaphor");
        vector<int> semaphor(cu.getPaddedNumAtoms());
        for (int i = 0; i < cu.getPaddedNumAtoms(); i++) semaphor[i] = 0;
        Semaphor->upload(semaphor);
        if (volScalingFactor) delete volScalingFactor;
        volScalingFactor = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "volScalingFactor");
        if (BornRadius) delete BornRadius;
        BornRadius = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "BornRadius");
        if (invBornRadius) delete invBornRadius;
        invBornRadius = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "invBornRadius");
        if (invBornRadius_fp) delete invBornRadius_fp;
        invBornRadius_fp = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "invBornRadius_fp");
        if (GBDerY) delete GBDerY;
        GBDerY = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(),
                                          "GBDerY"); //Y intermediate variable for Born radius-dependent GB derivative
        if (GBDerBrU) delete GBDerBrU;
        GBDerBrU = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(),
                                            "GBDerBrU"); //bru variable for Born radius-dependent GB derivative
        if (GBDerU) delete GBDerU;
        GBDerU = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(),
                                          "GBDerU"); //W variable for self-volume-dependent GB derivative
        if (VdWDerBrW) delete VdWDerBrW;
        VdWDerBrW = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(),
                                             "VdWDerBrW"); //brw variable for Born radius-dependent Van der Waals derivative
        if (VdWDerW) delete VdWDerW;
        VdWDerW = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(),
                                           "VdWDerW"); //W variable for self-volume-dependent vdW derivative

        //atomic parameters
        if (GaussianExponent) delete GaussianExponent;
        GaussianExponent = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "GaussianExponent");
        if (GaussianVolume) delete GaussianVolume;
        GaussianVolume = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "GaussianVolume");
        if (GaussianExponentLargeR) delete GaussianExponentLargeR;
        GaussianExponentLargeR = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "GaussianExponentLargeR");
        if (GaussianVolumeLargeR) delete GaussianVolumeLargeR;
        GaussianVolumeLargeR = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "GaussianVolumeLargeR");
        if (AtomicGamma) delete AtomicGamma;
        AtomicGamma = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "AtomicGamma");
        if (grad) delete grad;
        grad = CudaArray::create<float4>(cu, cu.getPaddedNumAtoms(), "grad");

    }

    // Reset tree kernel
    {
        map<string, string> defines;
        defines["FORCE_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);
        defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        defines["NUM_ATOMS_TREE"] = cu.intToString(gtree->total_atoms_in_tree);
        defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        defines["NUM_BLOCKS"] = cu.intToString(gtree->num_sections);
        defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);

        map<string, string> replacements;
        string file, kernel_name;
        CUmodule module;

        kernel_name = "resetTree";
        if (!hasCreatedKernels) {
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            file = cu.replaceStrings(CudaGKNPKernelSources::GVolResetTree, replacements);
            module = cu.createModule(file, defines);
            resetTreeKernel = cu.getKernel(module, kernel_name);
            // reset tree kernel
            if (verbose) cout << " done. " << endl;
        }

        // reset buffer kernel
        kernel_name = "resetBuffer";
        if (!hasCreatedKernels) {
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            resetBufferKernel= cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }


        // reset tree counters kernel
        kernel_name = "resetSelfVolumes";
        if (!hasCreatedKernels) {
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            resetSelfVolumesKernel= cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }
    }

    //Tree construction
    {
        CUmodule module;
        string kernel_name;

        //pass 1
        map<string, string> pairValueDefines;
        pairValueDefines["FORCE_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);
        pairValueDefines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        pairValueDefines["NUM_ATOMS_TREE"] = cu.intToString(gtree->total_atoms_in_tree);
        pairValueDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        pairValueDefines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
        pairValueDefines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
        pairValueDefines["OV_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);
        pairValueDefines["SMALL_VOLUME"] = "1.e-4";
        pairValueDefines["MAX_ORDER"] = cu.intToString(MAX_ORDER);

        if (useCutoff)
            pairValueDefines["USE_CUTOFF"] = "1";
        if (usePeriodic)
            pairValueDefines["USE_PERIODIC"] = "1";
        pairValueDefines["USE_EXCLUSIONS"] = "1";
        pairValueDefines["CUTOFF"] = cu.doubleToString(cutoffDistance);
        pairValueDefines["CUTOFF_SQUARED"] = cu.doubleToString(cutoffDistance * cutoffDistance);
        int numContexts = cu.getPlatformData().contexts.size();
        int numExclusionTiles = nb.getExclusionTiles().getSize();
        pairValueDefines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
        int startExclusionIndex = cu.getContextIndex() * numExclusionTiles / numContexts;
        int endExclusionIndex = (cu.getContextIndex() + 1) * numExclusionTiles / numContexts;
        pairValueDefines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
        pairValueDefines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);


        map<string, string> replacements;

        replacements["KFC"] = cu.doubleToString((double) KFC);
        replacements["VOLMIN0"] = cu.doubleToString((double) VOLMIN0);
        replacements["VOLMINA"] = cu.doubleToString((double) VOLMINA);
        replacements["VOLMINB"] = cu.doubleToString((double) VOLMINB);
        replacements["MIN_GVOL"] = cu.doubleToString((double) MIN_GVOL);

        replacements["ATOM_PARAMETER_DATA"] =
                "real4 g; \n"
                "real  v; \n"
                "real  gamma; \n"
                "int tree_pointer; \n";

        replacements["PARAMETER_ARGUMENTS"] = "";

        replacements["INIT_VARS"] = "";

        replacements["LOAD_ATOM1_PARAMETERS"] =
                "real a1 = global_gaussian_exponent[atom1]; \n"
                "real v1 = global_gaussian_volume[atom1];\n"
                "real gamma1 = global_atomic_gamma[atom1];\n";

        replacements["LOAD_LOCAL_PARAMETERS_FROM_1"] =
                "localData[localAtomIndex].g.w = a1;\n"
                "localData[localAtomIndex].v = v1;\n"
                "localData[localAtomIndex].gamma = gamma1;\n";


        replacements["LOAD_ATOM2_PARAMETERS"] =
                "real a2 = localData[localAtom2Index].g.w;\n"
                "real v2 = localData[localAtom2Index].v;\n"
                "real gamma2 = localData[localAtom2Index].gamma;\n";

        replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] =
                "localData[localAtomIndex].g.w = global_gaussian_exponent[j];\n"
                "localData[localAtomIndex].v = global_gaussian_volume[j];\n"
                "localData[localAtomIndex].gamma = global_atomic_gamma[j];\n"
                "localData[localAtomIndex].ov_count = 0;\n";




        //tree locks were used in the 2-body tree construction kernel. no more
        replacements["ACQUIRE_TREE_LOCK"] = "";
        replacements["RELEASE_TREE_LOCK"] = "";

        replacements["COMPUTE_INTERACTION_COUNT"] =
                "	real a12 = a1 + a2; \n"
                "	real deltai = 1./a12; \n"
                "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
                "	real dfp = df/PI; \n"
                "	real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       if(gvol > VolMinA ){ \n" //VolMin0?
                "          atomicAdd((int *)&ovChildrenCount[parent_slot], 1); \n"
                "       } \n";

        replacements["COMPUTE_INTERACTION_2COUNT"] =
                "	real a12 = a1 + a2; \n"
                "	real deltai = 1./a12; \n"
                "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
                "	real dfp = df/PI; \n"
                "	real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "	if(gvol > VolMin0 ){ \n"
                "          ov_count += 1; \n"
                "       } \n";

        replacements["COMPUTE_INTERACTION_GVOLONLY"] =
                "	real a12 = a1 + a2; \n"
                "       real df = a1*a2/a12; \n"
                "       real ef = exp(-df*r2); \n"
                "	real dfp = df/PI; \n"
                "       real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n";

        replacements["COMPUTE_INTERACTION_OTHER"] =
                "         real a12 = a1 + a2; \n"
                "         real df = a1*a2/a12; \n"
                "         real dgvol = -2.0f*df*gvol; \n"
                "         real dgvolv = v1 > 0 ? gvol/v1 : 0; \n"
                "	  //real4 c12 = (a1*posq1 + a2*posq2)/a12; \n"
                "       real4 c12 = make_real4((a1*posq1.x + a2*posq2.x)/a12, (a1*posq1.y + a2*posq2.y)/a12, (a1*posq1.z + a2*posq2.z)/a12, (a1*posq1.w + a2*posq2.w)/a12); \n"
                "         //switching function \n"
                "         real s = 0, sp = 0; \n"
                "         if(gvol > VolMinB ){ \n"
                "             s = 1.0f; \n"
                "             sp = 0.0f; \n"
                "         }else{ \n"
                "             real swd = 1.f/( VolMinB - VolMinA ); \n"
                "             real swu = (gvol - VolMinA)*swd; \n"
                "             real swu2 = swu*swu; \n"
                "             real swu3 = swu*swu2; \n"
                "             s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "             sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
                "         }\n"
                "         // switching function end \n"
                "	  real sfp = sp*gvol + s; \n";


        replacements["COMPUTE_INTERACTION_STORE1"] =
                "	real a12 = a1 + a2; \n"
                "	real deltai = 1./a12; \n"
                "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
                "	real dfp = df/PI; \n"
                "	real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       if(gvol > VolMinA){\n"
                "         real dgvol = -2.0f*df*gvol; \n"
                "         real dgvolv = v1 > 0 ? gvol/v1 : 0; \n"
                "	       //real4 c12 = deltai*(a1*posq1 + a2*posq2); \n"
                "         real4 c12 = make_real4(deltai*(a1*posq1.x + a2*posq2.x), deltai*(a1*posq1.y + a2*posq2.y), deltai*(a1*posq1.z + a2*posq2.z), deltai*(a1*posq1.w + a2*posq2.w)); \n"
                "         //switching function \n"
                "         real s = 0, sp = 0; \n"
                "         if(gvol > VolMinB ){ \n"
                "             s = 1.0f; \n"
                "             sp = 0.0f; \n"
                "         }else{ \n"
                "             real swd = 1.f/( VolMinB - VolMinA ); \n"
                "             real swu = (gvol - VolMinA)*swd; \n"
                "             real swu2 = swu*swu; \n"
                "             real swu3 = swu*swu2; \n"
                "             s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "             sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
                "         }\n"
                "         // switching function end \n"
                "	  real sfp = sp*gvol + s; \n"
                "         /* at this point have:\n"
                "	     1. gvol: overlap  between atom1 and atom2\n"
                "	     2. a12: gaussian exponent of overlap\n"
                "	     3. v12=gvol: volume of overlap\n"
                "	     4. c12: gaussian center of overlap\n"
                "	     These, together with atom2 (last_atom) are entered into the tree for atom 1 if\n"
                "	     volume is large enough.\n"
                "	 */\n"
                "        int endslot, children_count;\n"
                "        if(s*gvol > SMALL_VOLUME){ \n"
                "          //use top counter \n"
                "          children_count = atomicAdd(&ovChildrenCountTop[parent_slot], 1); \n"
                "          endslot = parent_children_start + children_count; \n"
                "        }else{ \n"
                "          //use bottom counter \n"
                "          children_count = atomicAdd(&ovChildrenCountBottom[parent_slot], 1); \n"
                "          endslot = parent_children_start + ovChildrenCount[parent_slot] - children_count - 1; \n"
                "        }\n"
                "        ovLevel[endslot] = 2; //two-body\n"
                "	 ovVolume[endslot] = gvol;\n"
                "        ovVsp[endslot] = s; \n"
                "        ovVSfp[endslot] = sfp; \n"
                "	 ovGamma1i[endslot] = gamma1 + gamma2;\n"
                "	 ovLastAtom[endslot] = child_atom;\n"
                "	 ovRootIndex[endslot] = parent_slot;\n"
                "	 ovChildrenStartIndex[endslot] = -1;\n"
                "	 ovChildrenCount[endslot] = 0;\n"
                "	 //ovG[endslot] = (real4)(c12.xyz, a12);\n"
                "        //ovDV1[endslot] = (real4)(-delta.xyz*dgvol,dgvolv);\n"
                "    ovG[endslot] = make_real4(c12.x, c12.y, c12.z, a12);\n"
                "        ovDV1[endslot] = make_real4(-delta.x*dgvol, -delta.y*dgvol, -delta.z*dgvol, dgvolv);\n"
                "      }\n";


        replacements["COMPUTE_INTERACTION_STORE2"] =
                "	real a12 = a1 + a2; \n"
                "	real deltai = 1./a12; \n"
                "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
                "	real dfp = df/PI; \n"
                "	real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       if(gvol > VolMinA){\n"
                "         real dgvol = -2.0f*df*gvol; \n"
                "         real dgvolv = v1 > 0 ? gvol/v1 : 0; \n"
                "	  //real4 c12 = deltai*(a1*posq1 + a2*posq2); \n"
                "     real4 c12 = make_real4(deltai*(a1*posq1.x + a2*posq2.x), deltai*(a1*posq1.y + a2*posq2.y), deltai*(a1*posq1.z + a2*posq2.z), deltai*(a1*posq1.w + a2*posq2.w)); \n"
                "         //switching function \n"
                "         real s = 0, sp = 0; \n"
                "         if(gvol > VolMinB ){ \n"
                "             s = 1.0f; \n"
                "             sp = 0.0f; \n"
                "         }else{ \n"
                "             real swd = 1.f/( VolMinB - VolMinA ); \n"
                "             real swu = (gvol - VolMinA)*swd; \n"
                "             real swu2 = swu*swu; \n"
                "             real swu3 = swu*swu2; \n"
                "             s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "             sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
                "         }\n"
                "         // switching function end \n"
                "	  real sfp = sp*gvol + s; \n"
                "	  /* at this point have:\n"
                "	     1. gvol: overlap  between atom1 and atom2\n"
                "	     2. a12: gaussian exponent of overlap\n"
                "	     3. v12=gvol: volume of overlap\n"
                "	     4. c12: gaussian center of overlap\n"
                "	     These, together with atom2 (last_atom) are entered into the tree for atom 1 if\n"
                "	     volume is large enough.\n"
                "	 */\n"
                "        int endslot, children_count;\n"
                "        if(s*gvol > SMALL_VOLUME){ \n"
                "          //use top counter \n"
                "          children_count = ovChildrenCountTop[slot]++; \n"
                "          endslot = ovChildrenStartIndex[slot] + children_count; \n"
                "        }else{ \n"
                "          //use bottom counter \n"
                "          children_count = ovChildrenCountBottom[slot]++; \n"
                "          endslot = ovChildrenStartIndex[slot] + ovChildrenCount[slot] - children_count - 1; \n"
                "        }\n"
                "	  ovLevel[endslot] = level + 1; //two-body\n"
                "	  ovVolume[endslot] = gvol;\n"
                "         ovVsp[endslot] = s; \n"
                "         ovVSfp[endslot] = sfp; \n"
                "	  ovGamma1i[endslot] = gamma1 + gamma2;\n"
                "	  ovLastAtom[endslot] = atom2;\n"
                "	  ovRootIndex[endslot] = slot;\n"
                "	  ovChildrenStartIndex[endslot] = -1;\n"
                "	  ovChildrenCount[endslot] = 0;\n"
                "	  //ovG[endslot] = (real4)(c12.xyz, a12);\n"
                "         //ovDV1[endslot] = (real4)(-delta.xyz*dgvol,dgvolv);\n"
                "     ovG[endslot] = make_real4(c12.x, c12.y, c12.z, a12);\n"
                "         ovDV1[endslot] = make_real4(-delta.x*dgvol, -delta.y*dgvol, -delta.z*dgvol, dgvolv); \n"
                "         ovProcessedFlag[endslot] = 0;\n"
                "         ovOKtoProcessFlag[endslot] = 1;\n"
                "       }\n";


        replacements["COMPUTE_INTERACTION_RESCAN"] =
                "	real a12 = a1 + a2; \n"
                "	real deltai = 1./a12; \n"
                "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
                "	real dfp = df/PI; \n"
                "	real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       real dgvol = -2.0f*df*gvol; \n"
                "       real dgvolv = v1 > 0 ? gvol/v1 : 0; \n"
                "       //real4 c12 = deltai*(a1*posq1 + a2*posq2); \n"
                "       real4 c12 = make_real4(deltai*(a1*posq1.x + a2*posq2.x), deltai*(a1*posq1.y + a2*posq2.y), deltai*(a1*posq1.z + a2*posq2.z), deltai*(a1*posq1.w + a2*posq2.w)); \n"
                "       //switching function \n"
                "       real s = 0, sp = 0; \n"
                "       if(gvol > VolMinB ){ \n"
                "           s = 1.0f; \n"
                "           sp = 0.0f; \n"
                "       }else{ \n"
                "           real swd = 1.f/( VolMinB - VolMinA ); \n"
                "           real swu = (gvol - VolMinA)*swd; \n"
                "           real swu2 = swu*swu; \n"
                "           real swu3 = swu*swu2; \n"
                "           s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "           sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
                "       }\n"
                "       // switching function end \n"
                "       real sfp = sp*gvol + s; \n"
                "       ovVolume[slot] = gvol;\n"
                "       ovVsp[slot] = s; \n"
                "       ovVSfp[slot] = sfp; \n"
                "       //ovG[slot] = (real4)(c12.xyz, a12);\n"
                "       //ovDV1[slot] = (real4)(-delta.xyz*dgvol,dgvolv);\n"
                "       ovG[slot] = make_real4(c12.x, c12.y, c12.z, a12);\n"
                "       ovDV1[slot] = make_real4(-delta.x*dgvol, -delta.y*dgvol, -delta.z*dgvol, dgvolv); \n";


        int reset_tree_size;

        string InitOverlapTreeSrc;

        kernel_name = "InitOverlapTree_1body";//large radii
        if (!hasCreatedKernels) {
            InitOverlapTreeSrc = cu.replaceStrings(CudaGKNPKernelSources::GVolOverlapTree, replacements);

            replacements["KERNEL_NAME"] = kernel_name;

            if (verbose) cout << "compiling GVolOverlapTree ...";
            module = cu.createModule(InitOverlapTreeSrc, pairValueDefines);
            if (verbose) cout << " done. " << endl;

            if (verbose) cout << "compiling " << kernel_name << " ... ";
            InitOverlapTreeKernel_1body_1 = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }
        reset_tree_size = 1;

        if (!hasCreatedKernels) {
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            module = cu.createModule(InitOverlapTreeSrc, pairValueDefines);
            if (verbose) cout << " done. " << endl;
            InitOverlapTreeKernel_1body_2 = cu.getKernel(module, kernel_name);
        }
        reset_tree_size = 0;

        kernel_name = "InitOverlapTreeCount";
        replacements["KERNEL_NAME"] = kernel_name;

        if (!hasCreatedKernels) {
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            InitOverlapTreeCountKernel = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }


        if (!hasCreatedKernels) {
            kernel_name = "reduceovCountBuffer";
            replacements["KERNEL_NAME"] = kernel_name;
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            reduceovCountBufferKernel = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }


        if (!hasCreatedKernels) {
            kernel_name = "InitOverlapTree";
            replacements["KERNEL_NAME"] = kernel_name;
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            InitOverlapTreeKernel = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }

        if (!hasCreatedKernels) {
            kernel_name = "resetComputeOverlapTree";
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            module = cu.createModule(InitOverlapTreeSrc, pairValueDefines);
            resetComputeOverlapTreeKernel = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }


        //pass 2 (1 pass kernel)
        if (!hasCreatedKernels) {
            kernel_name = "ComputeOverlapTree_1pass";
            replacements["KERNEL_NAME"] = kernel_name;
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            ComputeOverlapTree_1passKernel = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }

        //2-body volumes sort kernel
        if(!hasCreatedKernels) {
            kernel_name = "SortOverlapTree2body";
            replacements["KERNEL_NAME"] = kernel_name;
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            SortOverlapTree2bodyKernel = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }

        //rescan kernels
        if(!hasCreatedKernels) {
            kernel_name = "ResetRescanOverlapTree";
            replacements["KERNEL_NAME"] = kernel_name;
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            ResetRescanOverlapTreeKernel = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }

        if (!hasCreatedKernels) {
            kernel_name = "InitRescanOverlapTree";
            replacements["KERNEL_NAME"] = kernel_name;
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            InitRescanOverlapTreeKernel = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }

        //propagates atomic parameters (radii, gammas, etc) from the top to the bottom
        //of the overlap tree, recomputes overlap volumes as it goes
        if (!hasCreatedKernels) {
            kernel_name = "RescanOverlapTree";
            replacements["KERNEL_NAME"] = kernel_name;
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            RescanOverlapTreeKernel = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }


        //seeds tree with van der Waals + GB gamma parameters
        if (!hasCreatedKernels) {
            kernel_name = "InitOverlapTreeGammas_1body";
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            InitOverlapTreeGammasKernel_1body_W = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }

        //Same as RescanOverlapTree above:
        //propagates van der Waals + GB gamma atomic parameters from the top to the bottom
        //of the overlap tree,
        //it *does not* recompute overlap volumes
        //  used to prep calculations of volume derivatives of van der Waals energy
        if (!hasCreatedKernels) {
            kernel_name = "RescanOverlapTreeGammas";
            replacements["KERNEL_NAME"] = kernel_name;
            if (verbose) cout << "compiling " << kernel_name << "... ";
            RescanOverlapTreeGammasKernel_W = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }

    }

    //Self volumes kernel
    {

        map<string, string> defines;
        defines["FORCE_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);
        defines["NUM_ATOMS_TREE"] = cu.intToString(gtree->total_atoms_in_tree);
        defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
        defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
        defines["OV_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);

        map<string, string> replacements;
        CUmodule module;
        string kernel_name;
        string file;

        kernel_name = "computeSelfVolumes";
        if (!hasCreatedKernels) {
            file = cu.replaceStrings(CudaGKNPKernelSources::GVolSelfVolume, replacements);
            if (verbose) cout << "compiling file GVolSelfVolume.cu ... ";
            defines["DO_SELF_VOLUMES"] = "1";

            module = cu.createModule(file, defines);
            //accumulates self volumes and volume energy function (and forces)
            //with the energy-per-unit-volume parameters (Gamma1i) currently loaded into tree
            if (verbose) cout << "compiling kernel " << kernel_name << " ... ";
            computeSelfVolumesKernel = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }
    }

    //Self volumes reduction kernel (pass 2)
    {
        map<string, string> defines;
        defines["FORCE_WORK_GROUP_SIZE"] = cu.intToString(ov_work_group_size);
        defines["NUM_ATOMS_TREE"] = cu.intToString(gtree->total_atoms_in_tree);
        defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
        defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
        defines["NTILES_IN_BLOCK"] = "1";//cu.intToString(ov_work_group_size/CudaContext::TileSize);


        map<string, string> replacements;
        string kernel_name, file;
        CUmodule module;

        kernel_name = "reduceSelfVolumes_buffer";
        if (!hasCreatedKernels) {
            file = CudaGKNPKernelSources::GVolReduceTree;
            if (verbose) cout << "compiling file GVolReduceTree.cu ... ";

            module = cu.createModule(file, defines);

            if (verbose) cout << "compiling " << kernel_name << " ... ";
            reduceSelfVolumesKernel_buffer = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }


        kernel_name = "updateSelfVolumesForces";
        if (!hasCreatedKernels) {
            if (verbose) cout << "compiling " << kernel_name << " ... ";
            updateSelfVolumesForcesKernel = cu.getKernel(module, kernel_name);
            if (verbose) cout << " done. " << endl;
        }
    }
}

double GKNPPlugin::CudaCalcGKNPForceKernel::executeGVolSA(ContextImpl &context, bool includeForces, bool includeEnergy) {
    CudaNonbondedUtilities &nb = cu.getNonbondedUtilities();
    //bool useLong = cu.getSupports64BitGlobalAtomics();
    bool verbose = verbose_level > 0;
    niterations += 1;

    if (verbose) cout << "Executing GVolSA" << endl;

    bool nb_reassign = false;
    if (useCutoff) {
        if (maxTiles < nb.getInteractingTiles().getSize()) {
            maxTiles = nb.getInteractingTiles().getSize();
            nb_reassign = true;
            if (verbose)
                cout << "Reassigning neighbor list ..." << endl;
        }
    }

    unsigned int num_sections = gtree->num_sections;
    unsigned int paddedNumAtoms = cu.getPaddedNumAtoms();
    unsigned int numAtoms = cu.getNumAtoms();
    //------------------------------------------------------------------------------------------------------------
    // Tree construction (large radii)
    //
    //Execute resetTreeKernel
    {if (verbose_level > 1) cout << "Executing resetTreeKernel" << endl;
    //here workgroups cycle through tree sections to reset the tree section


    void *resetTreeKernelArgs[] = {&num_sections,
                                   &gtree->ovTreePointer->getDevicePointer(),
                                   &gtree->ovAtomTreePointer->getDevicePointer(),
                                   &gtree->ovAtomTreeSize->getDevicePointer(),
                                   &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                   &gtree->ovLevel->getDevicePointer(),
                                   &gtree->ovVolume->getDevicePointer(),
                                   &gtree->ovVsp->getDevicePointer(),
                                   &gtree->ovVSfp->getDevicePointer(),
                                   &gtree->ovSelfVolume->getDevicePointer(),
                                   &gtree->ovVolEnergy->getDevicePointer(),
                                   &gtree->ovLastAtom->getDevicePointer(),
                                   &gtree->ovRootIndex->getDevicePointer(),
                                   &gtree->ovChildrenStartIndex->getDevicePointer(),
                                   &gtree->ovChildrenCount->getDevicePointer(),
                                   &gtree->ovDV1->getDevicePointer(),
                                   &gtree->ovDV2->getDevicePointer(),
                                   &gtree->ovProcessedFlag->getDevicePointer(),
                                   &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                   &gtree->ovChildrenReported->getDevicePointer(),
                                   &gtree->ovAtomTreeLock->getDevicePointer(),
                                   &gtree->NIterations->getDevicePointer()
                                  };
    cu.executeKernel(resetTreeKernel, resetTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute resetBufferKernel
    {if (verbose_level > 1) cout << "Executing resetBufferKernel" << endl;
    // resets either ovAtomBuffer and long energy buffer
    void *resetBufferKernelArgs[] = {&paddedNumAtoms,
                                     &num_sections,
                                     &gtree->ovAtomBuffer->getDevicePointer(),
                                     &gtree->selfVolumeBuffer->getDevicePointer(),
                                     &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                     &gtree->gradBuffers_long->getDevicePointer()};
    cu.executeKernel(resetBufferKernel, resetBufferKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute InitOverlapTreeKernel_1body_1
    {if (verbose_level > 1) cout << "Executing InitOverlapTreeKernel_1body_1" << endl;
    //fills up tree with 1-body overlaps
    unsigned int reset_tree_size =1;
    void *InitOverlapTreeKernel_1body_1Args[] = {&paddedNumAtoms,
                                                 &num_sections,
                                                 &reset_tree_size,
                                                 &gtree->ovTreePointer->getDevicePointer(),
                                                 &gtree->ovNumAtomsInTree->getDevicePointer(),
                                                 &gtree->ovFirstAtom->getDevicePointer(),
                                                 &gtree->ovAtomTreeSize->getDevicePointer(),
                                                 &gtree->NIterations->getDevicePointer(),
                                                 &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                 &gtree->ovAtomTreePointer->getDevicePointer(),
                                                 &cu.getPosq().getDevicePointer(),
                                                 &radiusParam1->getDevicePointer(),
                                                 &gammaParam1->getDevicePointer(),
                                                 &ishydrogenParam->getDevicePointer(),
                                                 &GaussianExponent->getDevicePointer(),
                                                 &GaussianVolume->getDevicePointer(),
                                                 &AtomicGamma->getDevicePointer(),
                                                 &gtree->ovLevel->getDevicePointer(),
                                                 &gtree->ovVolume->getDevicePointer(),
                                                 &gtree->ovVsp->getDevicePointer(),
                                                 &gtree->ovVSfp->getDevicePointer(),
                                                 &gtree->ovGamma1i->getDevicePointer(),
                                                 &gtree->ovG->getDevicePointer(),
                                                 &gtree->ovDV1->getDevicePointer(),
                                                 &gtree->ovLastAtom->getDevicePointer(),
                                                 &gtree->ovRootIndex->getDevicePointer(),
                                                 &gtree->ovChildrenStartIndex->getDevicePointer(),
                                                 &gtree->ovChildrenCount->getDevicePointer()
    };
//    int threads = ov_work_group_size * num_compute_units;
//    int blockSize = ov_work_group_size;
//    int sharedSize=0;
//    int numThreadBlocks=cu.getNumThreadBlocks();
//    int gridSize = std::min((threads+blockSize-1)/blockSize, numThreadBlocks);
//    CUresult result = CUDAAPI::cuLaunchKernel(InitOverlapTreeKernel_1body_1, gridSize, 1, 1, blockSize, 1, 1, sharedSize, cu.getCurrentStream(), InitOverlapTreeKernel_1body_1Args, NULL);
    cu.executeKernel(InitOverlapTreeKernel_1body_1, InitOverlapTreeKernel_1body_1Args, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute InitOverlapTreeCountKernel
    {if (verbose_level > 1) cout << "Executing InitOverlapTreeCountKernel" << endl;
    // compute numbers of 2-body overlaps, that is children counts of 1-body overlaps
//    if (nb_reassign) {
//        int index = InitOverlapTreeCountKernel_first_nbarg;
//        CUfunction kernel = InitOverlapTreeCountKernel;
//        kernel.setArg<cl::Buffer>(index++, nb.getInteractingTiles().getDeviceBuffer());
//        kernel.setArg<cl::Buffer>(index++, nb.getInteractionCount().getDeviceBuffer());
//        kernel.setArg<cl::Buffer>(index++, nb.getInteractingAtoms().getDeviceBuffer());
//        kernel.setArg<unsigned int>(index++, nb.getInteractingTiles().getSize());
//        kernel.setArg<cl::Buffer>(index++, nb.getExclusionTiles().getDeviceBuffer());
//    }

        unsigned int interactingTileSize = nb.getInteractingTiles().getSize();
        void *InitOverlapTreeCountKernelCutoffArgs[] = {&gtree->ovAtomTreePointer->getDevicePointer(),
                                                        &cu.getPosq().getDevicePointer(),
                                                        &GaussianExponent->getDevicePointer(),
                                                        &GaussianVolume->getDevicePointer(),
                                                        &nb.getInteractingTiles().getDevicePointer(),
                                                        &nb.getInteractionCount().getDevicePointer(),
                                                        &nb.getInteractingAtoms().getDevicePointer(),
                                                        &interactingTileSize,
                                                        &nb.getExclusionTiles().getDevicePointer(),
                                                        &gtree->ovChildrenCount->getDevicePointer()};

        unsigned int numAtomBlocks = (cu.getNumAtomBlocks() * (cu.getNumAtomBlocks() + 1) / 2);
        void *InitOverlapTreeCountKernelArgs[] = {&gtree->ovAtomTreePointer->getDevicePointer(),
                                                  &cu.getPosq().getDevicePointer(),
                                                  &GaussianExponent->getDevicePointer(),
                                                  &GaussianVolume->getDevicePointer(),
                                                  &numAtomBlocks,
                                                  &gtree->ovChildrenCount->getDevicePointer()};
    if(useCutoff){
        cu.executeKernel(InitOverlapTreeCountKernel, InitOverlapTreeCountKernelCutoffArgs, ov_work_group_size * num_compute_units, ov_work_group_size);
    }
    else{
        cu.executeKernel(InitOverlapTreeCountKernel, InitOverlapTreeCountKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);
    }
    }

    //Execute reduceovCountBufferKernel
    {if (verbose_level > 1) cout << "Executing reduceovCountBufferKernel" << endl;
    // do a prefix sum of 2-body counts to compute children start indexes to store 2-body overlaps computed by InitOverlapTreeKernel below
    void *reduceovCountBufferKernelArgs[] = {&num_sections,
                                             &gtree->ovTreePointer->getDevicePointer(),
                                             &gtree->ovAtomTreePointer->getDevicePointer(),
                                             &gtree->ovAtomTreeSize->getDevicePointer(),
                                             &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                             &gtree->ovChildrenStartIndex->getDevicePointer(),
                                             &gtree->ovChildrenCount->getDevicePointer(),
                                             &gtree->ovChildrenCountTop->getDevicePointer(),
                                             &gtree->ovChildrenCountBottom->getDevicePointer(),
                                             &PanicButton->getDevicePointer()};
    cu.executeKernel(reduceovCountBufferKernel, reduceovCountBufferKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    /*if (verbose_level > 4) {
        float self_volume = 0.0;
        vector<float> self_volumes(gtree->total_tree_size);
        vector<float> volumes(gtree->total_tree_size);
        vector<float> energies(gtree->total_tree_size);
        vector<float> gammas(gtree->total_tree_size);
        vector<int> last_atom(gtree->total_tree_size);
        vector<int> level(gtree->total_tree_size);
        vector<int> parent(gtree->total_tree_size);
        vector<int> children_start_index(gtree->total_tree_size);
        vector<int> children_count(gtree->total_tree_size);
        vector<int> children_reported(gtree->total_tree_size);
        vector<float4> g(gtree->total_tree_size);
        vector<float4> dv2(gtree->total_tree_size);
        vector<float4> dv1(gtree->total_tree_size);
        vector<float> sfp(gtree->total_tree_size);
        vector<int> size(gtree->num_sections);
        vector<int> tree_pointer_t(gtree->num_sections);
        vector<int> processed(gtree->total_tree_size);
        vector<int> oktoprocess(gtree->total_tree_size);


        gtree->ovSelfVolume->download(self_volumes);
        gtree->ovVolume->download(volumes);
        gtree->ovVolEnergy->download(energies);
        gtree->ovLevel->download(level);
        gtree->ovLastAtom->download(last_atom);
        gtree->ovRootIndex->download(parent);
        gtree->ovChildrenStartIndex->download(children_start_index);
        gtree->ovChildrenCount->download(children_count);
        gtree->ovChildrenReported->download(children_reported);
        gtree->ovG->download(g);
        gtree->ovGamma1i->download(gammas);
        gtree->ovDV1->download(dv1);
        gtree->ovDV2->download(dv2);
        gtree->ovVSfp->download(sfp);
        gtree->ovAtomTreeSize->download(size);
        gtree->ovTreePointer->download(tree_pointer_t);
        gtree->ovProcessedFlag->download(processed);
        gtree->ovOKtoProcessFlag->download(oktoprocess);


        std::cout << "Tree:" << std::endl;
        for (int section = 0; section < gtree->num_sections; section++) {
            std::cout << "Tree for sections: " << section << " " << " size= " << size[section] << std::endl;
            int pp = tree_pointer_t[section];
            int np = gtree->padded_tree_size[section];
            //self_volume += self_volumes[pp];
            std::cout
                    << "slot level LastAtom parent ChStart ChCount SelfV V gamma Energy a x y z dedx dedy dedz sfp processed ok2process children_reported"
                    << endl;
            for (int i = pp; i < pp + np; i++) {
                int maxprint = pp + 1024;
                if (i < maxprint) {
                    std::cout << std::setprecision(4) << std::setw(6) << i << " " << std::setw(7) << (int) level[i]
                              << " " << std::setw(7) << (int) last_atom[i] << " " << std::setw(7) << (int) parent[i]
                              << " " << std::setw(7) << (int) children_start_index[i] << " " << std::setw(7)
                              << (int) children_count[i] << " " << std::setw(15) << (float) self_volumes[i] << " "
                              << std::setw(10) << (float) volumes[i] << " " << std::setw(10) << (float) gammas[i] << " "
                              << std::setw(10) << (float) energies[i] << " " << std::setw(10) << g[i].w << " "
                              << std::setw(10) << g[i].x << " " << std::setw(10) << g[i].y << " " << std::setw(10)
                              << g[i].z << " " << std::setw(10) << dv2[i].x << " " << std::setw(10) << dv2[i].y << " "
                              << std::setw(10) << dv2[i].z << " " << std::setw(10) << sfp[i] << " " << processed[i]
                              << " " << oktoprocess[i] << " " << children_reported[i] << std::endl;
                }
            }
        }
        //std::cout << "Volume (from self volumes):" << self_volume <<std::endl;
    }*/

    //Execute InitOverlapTreeKernel
    {if (verbose_level > 1) cout << "Executing InitOverlapTreeKernel" << endl;
//    if (nb_reassign) {
//        int index = InitOverlapTreeKernel_first_nbarg;
//        CUfunction kernel = InitOverlapTreeKernel;
//        kernel.setArg<cl::Buffer>(index++, nb.getInteractingTiles().getDeviceBuffer());
//        kernel.setArg<cl::Buffer>(index++, nb.getInteractionCount().getDeviceBuffer());
//        kernel.setArg<cl::Buffer>(index++, nb.getInteractingAtoms().getDeviceBuffer());
//        kernel.setArg<unsigned int>(index++, nb.getInteractingTiles().getSize());
//        kernel.setArg<cl::Buffer>(index++, nb.getExclusionTiles().getDeviceBuffer());
//    }

    unsigned int interactingTileSize = nb.getInteractingTiles().getSize();
    unsigned int numAtomBlocks = (cu.getNumAtomBlocks() * (cu.getNumAtomBlocks() + 1) / 2);
    void *InitOverlapTreeKernelCutoffArgs[] = {&gtree->ovAtomTreePointer->getDevicePointer(),
                                               &gtree->ovAtomTreeSize->getDevicePointer(),
                                               &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                               &cu.getPosq().getDevicePointer(),
                                               &GaussianExponent->getDevicePointer(),
                                               &GaussianVolume->getDevicePointer(),
                                               &AtomicGamma->getDevicePointer(),
                                               &nb.getInteractingTiles().getDevicePointer(),
                                               &nb.getInteractionCount().getDevicePointer(),
                                               &nb.getInteractingAtoms().getDevicePointer(),
                                               &interactingTileSize,
                                               &nb.getExclusionTiles().getDevicePointer(),
                                               &gtree->ovLevel->getDevicePointer(),
                                               &gtree->ovVolume->getDevicePointer(),
                                               &gtree->ovVsp->getDevicePointer(),
                                               &gtree->ovVSfp->getDevicePointer(),
                                               &gtree->ovGamma1i->getDevicePointer(),
                                               &gtree->ovG->getDevicePointer(),
                                               &gtree->ovDV1->getDevicePointer(),
                                               &gtree->ovLastAtom->getDevicePointer(),
                                               &gtree->ovRootIndex->getDevicePointer(),
                                               &gtree->ovChildrenStartIndex->getDevicePointer(),
                                               &gtree->ovChildrenCount->getDevicePointer(),
                                               &gtree->ovChildrenCountTop->getDevicePointer(),
                                               &gtree->ovChildrenCountBottom->getDevicePointer(),
                                               &PanicButton->getDevicePointer()};

    void *InitOverlapTreeKernelArgs[] = {&gtree->ovAtomTreePointer->getDevicePointer(),
                                         &gtree->ovAtomTreeSize->getDevicePointer(),
                                         &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                         &cu.getPosq().getDevicePointer(),
                                         &GaussianExponent->getDevicePointer(),
                                         &GaussianVolume->getDevicePointer(),
                                         &AtomicGamma->getDevicePointer(),
                                         &numAtomBlocks,
                                         &gtree->ovLevel->getDevicePointer(),
                                         &gtree->ovVolume->getDevicePointer(),
                                         &gtree->ovVsp->getDevicePointer(),
                                         &gtree->ovVSfp->getDevicePointer(),
                                         &gtree->ovGamma1i->getDevicePointer(),
                                         &gtree->ovG->getDevicePointer(),
                                         &gtree->ovDV1->getDevicePointer(),
                                         &gtree->ovLastAtom->getDevicePointer(),
                                         &gtree->ovRootIndex->getDevicePointer(),
                                         &gtree->ovChildrenStartIndex->getDevicePointer(),
                                         &gtree->ovChildrenCount->getDevicePointer(),
                                         &gtree->ovChildrenCountTop->getDevicePointer(),
                                         &gtree->ovChildrenCountBottom->getDevicePointer(),
                                         &PanicButton->getDevicePointer()};
    if(useCutoff){
        cu.executeKernel(InitOverlapTreeKernel, InitOverlapTreeKernelCutoffArgs, ov_work_group_size * num_compute_units, ov_work_group_size);
    }
    else{
        cu.executeKernel(InitOverlapTreeKernel, InitOverlapTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);
    }
    }

    //Execute resetComputeOverlapTreeKernel
    {if (verbose_level > 1) cout << "Executing resetComputeOverlapTreeKernel" << endl;
    void *resetComputeOverlapTreeKernelArgs[] = {&num_sections,
                                                 &gtree->ovTreePointer->getDevicePointer(),
                                                 &gtree->ovProcessedFlag->getDevicePointer(),
                                                 &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                                 &gtree->ovAtomTreeSize->getDevicePointer(),
                                                 &gtree->ovLevel->getDevicePointer()};
    cu.executeKernel(resetComputeOverlapTreeKernel, resetComputeOverlapTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute ComputeOverlapTree_1passKernel
    {if (verbose_level > 1) cout << "Executing ComputeOverlapTree_1passKernel" << endl;
    int temp_buffer_size = gtree->temp_buffer_size;
    void *ComputeOverlapTree_1passKernelArgs[] = {&num_sections,
                                                  &gtree->ovTreePointer->getDevicePointer(),
                                                  &gtree->ovAtomTreePointer->getDevicePointer(),
                                                  &gtree->ovAtomTreeSize->getDevicePointer(),
                                                  &gtree->NIterations->getDevicePointer(),
                                                  &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                  &gtree->ovAtomTreeLock->getDevicePointer(),
                                                  &cu.getPosq().getDevicePointer(),
                                                  &GaussianExponent->getDevicePointer(),
                                                  &GaussianVolume->getDevicePointer(),
                                                  &AtomicGamma->getDevicePointer(),
                                                  &gtree->ovLevel->getDevicePointer(),
                                                  &gtree->ovVolume->getDevicePointer(),
                                                  &gtree->ovVsp->getDevicePointer(),
                                                  &gtree->ovVSfp->getDevicePointer(),
                                                  &gtree->ovGamma1i->getDevicePointer(),
                                                  &gtree->ovG->getDevicePointer(),
                                                  &gtree->ovDV1->getDevicePointer(),
                                                  &gtree->ovLastAtom->getDevicePointer(),
                                                  &gtree->ovRootIndex->getDevicePointer(),
                                                  &gtree->ovChildrenStartIndex->getDevicePointer(),
                                                  &gtree->ovChildrenCount->getDevicePointer(),
                                                  &gtree->ovProcessedFlag->getDevicePointer(),
                                                  &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                                  &gtree->ovChildrenReported->getDevicePointer(),
                                                  &gtree->ovChildrenCountTop->getDevicePointer(),
                                                  &gtree->ovChildrenCountBottom->getDevicePointer(),
                                                  &temp_buffer_size,
                                                  &gtree->gvol_buffer_temp->getDevicePointer(),
                                                  &gtree->tree_pos_buffer_temp->getDevicePointer(),
                                                  &gtree->i_buffer_temp->getDevicePointer(),
                                                  &gtree->atomj_buffer_temp->getDevicePointer(),
                                                  &PanicButton->getDevicePointer()};
    cu.executeKernel(ComputeOverlapTree_1passKernel, ComputeOverlapTree_1passKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //TODO: Panic Button?
    //trigger non-blocking read of PanicButton, read it after next kernel below
//    cu.getQueue().enqueueReadBuffer(PanicButton->getDeviceBuffer(), CL_TRUE, 0, 2 * sizeof(int), &panic_button[0], NULL,
//                                    &downloadPanicButtonEvent);

    //------------------------------------------------------------------------------------------------------------


    //------------------------------------------------------------------------------------------------------------
    // Volume energy function 1 (large radii)
    //

    //Execute resetSelfVolumesKernel
    {if (verbose_level > 1) cout << "Executing resetSelfVolumesKernel" << endl;
    void *resetSelfVolumesArgs[] = {&num_sections,
                                    &gtree->ovTreePointer->getDevicePointer(),
                                    &gtree->ovAtomTreePointer->getDevicePointer(),
                                    &gtree->ovAtomTreeSize->getDevicePointer(),
                                    &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                    &gtree->ovChildrenStartIndex->getDevicePointer(),
                                    &gtree->ovChildrenCount->getDevicePointer(),
                                    &gtree->ovProcessedFlag->getDevicePointer(),
                                    &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                    &gtree->ovChildrenReported->getDevicePointer(),
                                    &PanicButton->getDevicePointer()};
    cu.executeKernel(resetSelfVolumesKernel, resetSelfVolumesArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //TODO: Panic Button?
    //check the result of the non-blocking read of PanicButton above
//    downloadPanicButtonEvent.wait();
//    if (panic_button[0] > 0) {
//        if (verbose) cout << "Error: Tree size exceeded(2)!" << endl;
//        hasInitializedKernels = false; //forces reinitialization
//        cu.setForcesValid(false); //invalidate forces
//
//        if (panic_button[1] > 0) {
//            if (verbose) cout << "Error: Temp Buffer exceeded(2)!" << endl;
//            gtree->hasExceededTempBuffer = true;//forces resizing of temp buffers
//        }
//
//        if (verbose) {
//            cout << "Tree sizes:" << endl;
//            vector<int> size(gtree->num_sections);
//            gtree->ovAtomTreeSize->download(size);
//            for (int section = 0; section < gtree->num_sections; section++) {
//                cout << size[section] << " ";
//            }
//            cout << endl;
//        }
//
//        return 0.0;
//    }

    //Execute computeSelfVolumesKernel
    {if (verbose_level > 1) cout << "Executing computeSelfVolumesKernel" << endl;
    void *computeSelfVolumesKernelArgs[] = {&num_sections,
                                            &gtree->ovTreePointer->getDevicePointer(),
                                            &gtree->ovAtomTreePointer->getDevicePointer(),
                                            &gtree->ovAtomTreeSize->getDevicePointer(),
                                            &gtree->NIterations->getDevicePointer(),
                                            &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                            &GaussianExponent->getDevicePointer(),
                                            &paddedNumAtoms,
                                            &gtree->ovLevel->getDevicePointer(),
                                            &gtree->ovVolume->getDevicePointer(),
                                            &gtree->ovVsp->getDevicePointer(),
                                            &gtree->ovVSfp->getDevicePointer(),
                                            &gtree->ovGamma1i->getDevicePointer(),
                                            &gtree->ovG->getDevicePointer(),
                                            &gtree->ovSelfVolume->getDevicePointer(),
                                            &gtree->ovVolEnergy->getDevicePointer(),
                                            &gtree->ovDV1->getDevicePointer(),
                                            &gtree->ovDV2->getDevicePointer(),
                                            &gtree->ovPF->getDevicePointer(),
                                            &gtree->ovLastAtom->getDevicePointer(),
                                            &gtree->ovRootIndex->getDevicePointer(),
                                            &gtree->ovChildrenStartIndex->getDevicePointer(),
                                            &gtree->ovChildrenCount->getDevicePointer(),
                                            &gtree->ovProcessedFlag->getDevicePointer(),
                                            &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                            &gtree->ovChildrenReported->getDevicePointer(),
                                            &gtree->ovAtomBuffer->getDevicePointer(),
                                            &gtree->gradBuffers_long->getDevicePointer(),
                                            &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                            &gtree->selfVolumeBuffer->getDevicePointer()};
    cu.executeKernel(computeSelfVolumesKernel, computeSelfVolumesKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute reduceSelfVolumesKernel_buffer
    {if (verbose_level > 1) cout << "Executing reduceSelfVolumesKernel_buffer" << endl;
    void *reduceSelfVolumesKernel_bufferArgs[] = {&numAtoms,
                                                  &paddedNumAtoms,
                                                  &num_sections,
                                                  &gtree->ovAtomTreePointer->getDevicePointer(),
                                                  &gtree->ovAtomBuffer->getDevicePointer(),
                                                  &gtree->gradBuffers_long->getDevicePointer(),
                                                  &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                                  &gtree->selfVolumeBuffer->getDevicePointer(),
                                                  &selfVolume->getDevicePointer(),
                                                  &GaussianVolume->getDevicePointer(),
                                                  &AtomicGamma->getDevicePointer(),
                                                  &grad->getDevicePointer()};
    cu.executeKernel(reduceSelfVolumesKernel_buffer, reduceSelfVolumesKernel_bufferArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute updateSelfVolumesForces
    {if (verbose_level > 1) cout << "Executing updateSelfVolumesForces" << endl << endl;
    int update_energy = 1;
    void *updateSelfVolumesForcesKernelArgs[] ={&update_energy,
                                               &numAtoms,
                                               &paddedNumAtoms,
                                               &gtree->ovAtomTreePointer->getDevicePointer(),
                                               &gtree->ovVolEnergy->getDevicePointer(),
                                               &grad->getDevicePointer(),
                                               &cu.getForce().getDevicePointer(),
                                               &cu.getEnergyBuffer().getDevicePointer()};
    cu.executeKernel(updateSelfVolumesForcesKernel, updateSelfVolumesForcesKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    if (false) {
        vector<int> size(gtree->num_sections);
        vector<int> niter(gtree->num_sections);
        gtree->ovAtomTreeSize->download(size);
        cout << "Sizes: ";
        for (int section = 0; section < gtree->num_sections; section++) {
            std::cout << size[section] << " ";
        }
        std::cout << endl;
        gtree->NIterations->download(niter);
        cout << "Niter: ";
        for (int section = 0; section < gtree->num_sections; section++) {
            std::cout << niter[section] << " ";
        }
        std::cout << endl;

    }

    double volume1;
    if (verbose_level > 1) {
        //print self volumes
        vector<float> self_volumes(cu.getPaddedNumAtoms());
        selfVolume->download(self_volumes);
        for (int i = 0; i < numParticles; i++) {
            printf("self_volume: %6.6f atom: %d\n", self_volumes[i], i);
            volume1+=self_volumes[i];
        }
    }



    vector<int> atom_pointer;
    vector<float> vol_energies;
    gtree->ovAtomTreePointer->download(atom_pointer);
    gtree->ovVolEnergy->download(vol_energies);
    double energy = 0;
    for (int i = 0; i < numParticles; i++) {
        int slot = atom_pointer[i];
        energy += vol_energies[slot];
        printf("vol_energies[%d]: %6.6f\n", slot, vol_energies[slot]);

    }
    //cout << "Volume 1: " << volume1/ANG3 << endl;
    //cout << "Volume Energy 1:" << energy << endl << endl;




    //------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------
    // Self volumes, volume scaling parameters,
    // volume energy function 2 (small radii), surface area cavity energy function
    //

    //seeds tree with "negative" gammas and reduced radii
    //Execute InitOverlapTreeKernel_1body_2
    {if (verbose_level > 1) cout << "Executing InitOverlapTreeKernel_1body_2 " << endl;
    int reset_tree_size = 0;
    void *InitOverlapTreeKernel_1body_2Args[] = {&paddedNumAtoms,
                                                 &num_sections,
                                                 &reset_tree_size,
                                                 &gtree->ovTreePointer->getDevicePointer(),
                                                 &gtree->ovNumAtomsInTree->getDevicePointer(),
                                                 &gtree->ovFirstAtom->getDevicePointer(),
                                                 &gtree->ovAtomTreeSize->getDevicePointer(),
                                                 &gtree->NIterations->getDevicePointer(),
                                                 &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                 &gtree->ovAtomTreePointer->getDevicePointer(),
                                                 &cu.getPosq().getDevicePointer(),
                                                 &radiusParam2->getDevicePointer(),
                                                 &gammaParam2->getDevicePointer(),
                                                 &ishydrogenParam->getDevicePointer(),
                                                 &GaussianExponent->getDevicePointer(),
                                                 &GaussianVolume->getDevicePointer(),
                                                 &AtomicGamma->getDevicePointer(),
                                                 &gtree->ovLevel->getDevicePointer(),
                                                 &gtree->ovVolume->getDevicePointer(),
                                                 &gtree->ovVsp->getDevicePointer(),
                                                 &gtree->ovVSfp->getDevicePointer(),
                                                 &gtree->ovGamma1i->getDevicePointer(),
                                                 &gtree->ovG->getDevicePointer(),
                                                 &gtree->ovDV1->getDevicePointer(),
                                                 &gtree->ovLastAtom->getDevicePointer(),
                                                 &gtree->ovRootIndex->getDevicePointer(),
                                                 &gtree->ovChildrenStartIndex->getDevicePointer(),
                                                 &gtree->ovChildrenCount->getDevicePointer()};
    cu.executeKernel(InitOverlapTreeKernel_1body_2, InitOverlapTreeKernel_1body_2Args, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute ResetRescanOverlapTreeKernel
    {if (verbose_level > 1) cout << "Executing ResetRescanOverlapTreeKernel" << endl;
    void *ResetRescanOverlapTreeKernelArgs[] = {&num_sections,
                                                &gtree->ovTreePointer->getDevicePointer(),
                                                &gtree->ovAtomTreePointer->getDevicePointer(),
                                                &gtree->ovAtomTreeSize->getDevicePointer(),
                                                &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                                &gtree->ovProcessedFlag->getDevicePointer(),
                                                &gtree->ovOKtoProcessFlag->getDevicePointer()};
    cu.executeKernel(ResetRescanOverlapTreeKernel, ResetRescanOverlapTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute InitRescanOverlapTreeKernel
    {if (verbose_level > 1) cout << "Executing InitRescanOverlapTreeKernel" << endl;
    void *InitRescanOverlapTreeKernelArgs[] = {&num_sections,
                                               &gtree->ovTreePointer->getDevicePointer(),
                                               &gtree->ovAtomTreeSize->getDevicePointer(),
                                               &gtree->ovProcessedFlag->getDevicePointer(),
                                               &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                               &gtree->ovLevel->getDevicePointer()};
    cu.executeKernel(InitRescanOverlapTreeKernel, InitRescanOverlapTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute RescanOverlapTreeKernel
    {if (verbose_level > 1) cout << "Executing RescanOverlapTreeKernel" << endl;
    void *RescanOverlapTreeKernelArgs[] = {&num_sections,
                                           &gtree->ovTreePointer->getDevicePointer(),
                                           &gtree->ovAtomTreePointer->getDevicePointer(),
                                           &gtree->ovAtomTreeSize->getDevicePointer(),
                                           &gtree->NIterations->getDevicePointer(),
                                           &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                           &gtree->ovAtomTreeLock->getDevicePointer(),
                                           &cu.getPosq().getDevicePointer(),
                                           &GaussianExponent->getDevicePointer(),
                                           &GaussianVolume->getDevicePointer(),
                                           &AtomicGamma->getDevicePointer(),
                                           &gtree->ovLevel->getDevicePointer(),
                                           &gtree->ovVolume->getDevicePointer(),
                                           &gtree->ovVsp->getDevicePointer(),
                                           &gtree->ovVSfp->getDevicePointer(),
                                           &gtree->ovGamma1i->getDevicePointer(),
                                           &gtree->ovG->getDevicePointer(),
                                           &gtree->ovDV1->getDevicePointer(),
                                           &gtree->ovLastAtom->getDevicePointer(),
                                           &gtree->ovRootIndex->getDevicePointer(),
                                           &gtree->ovChildrenStartIndex->getDevicePointer(),
                                           &gtree->ovChildrenCount->getDevicePointer(),
                                           &gtree->ovProcessedFlag->getDevicePointer(),
                                           &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                           &gtree->ovChildrenReported->getDevicePointer()};
    cu.executeKernel(RescanOverlapTreeKernel, RescanOverlapTreeKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute resetSelfVolumesKernel
    {if (verbose_level > 1) cout << "Executing resetSelfVolumesKernel" << endl;
    void *resetSelfVolumesArgs[] = {&num_sections,
                                    &gtree->ovTreePointer->getDevicePointer(),
                                    &gtree->ovAtomTreePointer->getDevicePointer(),
                                    &gtree->ovAtomTreeSize->getDevicePointer(),
                                    &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                    &gtree->ovChildrenStartIndex->getDevicePointer(),
                                    &gtree->ovChildrenCount->getDevicePointer(),
                                    &gtree->ovProcessedFlag->getDevicePointer(),
                                    &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                    &gtree->ovChildrenReported->getDevicePointer(),
                                    &PanicButton->getDevicePointer()};
    cu.executeKernel(resetSelfVolumesKernel, resetSelfVolumesArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    // zero self-volume accumulator
    //Executing resetBufferKernel
    {if (verbose_level > 1) cout << "Executing resetBufferKernel" << endl;
    void *resetBufferKernelArgs[] = {&paddedNumAtoms,
                                     &num_sections,
                                     &gtree->ovAtomBuffer->getDevicePointer(),
                                     &gtree->selfVolumeBuffer->getDevicePointer(),
                                     &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                     &gtree->gradBuffers_long->getDevicePointer()};
    cu.executeKernel(resetBufferKernel, resetBufferKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute computeSelfVolumesKernel
    {if (verbose_level > 1) cout << "Executing computeSelfVolumesKernel" << endl;
    void *computeSelfVolumesKernelArgs[] = {&num_sections,
                                            &gtree->ovTreePointer->getDevicePointer(),
                                            &gtree->ovAtomTreePointer->getDevicePointer(),
                                            &gtree->ovAtomTreeSize->getDevicePointer(),
                                            &gtree->NIterations->getDevicePointer(),
                                            &gtree->ovAtomTreePaddedSize->getDevicePointer(),
                                            &GaussianExponent->getDevicePointer(),
                                            &paddedNumAtoms,
                                            &gtree->ovLevel->getDevicePointer(),
                                            &gtree->ovVolume->getDevicePointer(),
                                            &gtree->ovVsp->getDevicePointer(),
                                            &gtree->ovVSfp->getDevicePointer(),
                                            &gtree->ovGamma1i->getDevicePointer(),
                                            &gtree->ovG->getDevicePointer(),
                                            &gtree->ovSelfVolume->getDevicePointer(),
                                            &gtree->ovVolEnergy->getDevicePointer(),
                                            &gtree->ovDV1->getDevicePointer(),
                                            &gtree->ovDV2->getDevicePointer(),
                                            &gtree->ovPF->getDevicePointer(),
                                            &gtree->ovLastAtom->getDevicePointer(),
                                            &gtree->ovRootIndex->getDevicePointer(),
                                            &gtree->ovChildrenStartIndex->getDevicePointer(),
                                            &gtree->ovChildrenCount->getDevicePointer(),
                                            &gtree->ovProcessedFlag->getDevicePointer(),
                                            &gtree->ovOKtoProcessFlag->getDevicePointer(),
                                            &gtree->ovChildrenReported->getDevicePointer(),
                                            &gtree->ovAtomBuffer->getDevicePointer(),
                                            &gtree->gradBuffers_long->getDevicePointer(),
                                            &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                            &gtree->selfVolumeBuffer->getDevicePointer()};
    cu.executeKernel(computeSelfVolumesKernel, computeSelfVolumesKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //update energyBuffer with volume energy 2
    //Execute reduceSelfVolumesKernel_buffer
    {if (verbose_level > 1) cout << "Executing reduceSelfVolumesKernel_buffer" << endl;
    void *reduceSelfVolumesKernel_bufferArgs[] = {&numAtoms,
                                                  &paddedNumAtoms,
                                                  &num_sections,
                                                  &gtree->ovAtomTreePointer->getDevicePointer(),
                                                  &gtree->ovAtomBuffer->getDevicePointer(),
                                                  &gtree->gradBuffers_long->getDevicePointer(),
                                                  &gtree->selfVolumeBuffer_long->getDevicePointer(),
                                                  &gtree->selfVolumeBuffer->getDevicePointer(),
                                                  &selfVolume->getDevicePointer(),
                                                  &GaussianVolume->getDevicePointer(),
                                                  &AtomicGamma->getDevicePointer(),
                                                  &grad->getDevicePointer()};
    cu.executeKernel(reduceSelfVolumesKernel_buffer, reduceSelfVolumesKernel_bufferArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    //Execute updateSelfVolumesForces
    {if (verbose_level > 1) cout << "Executing updateSelfVolumesForces" << endl << endl;
    int update_energy = 1;
    void *updateSelfVolumesForcesKernelArgs[] ={&update_energy,
                                               &numAtoms,
                                               &paddedNumAtoms,
                                               &gtree->ovAtomTreePointer->getDevicePointer(),
                                               &gtree->ovVolEnergy->getDevicePointer(),
                                               &grad->getDevicePointer(),
                                               &cu.getForce().getDevicePointer(),
                                               &cu.getEnergyBuffer().getDevicePointer()};
    cu.executeKernel(updateSelfVolumesForcesKernel, updateSelfVolumesForcesKernelArgs, ov_work_group_size * num_compute_units, ov_work_group_size);}

    if (verbose_level > 4) {
        float self_volume = 0.0;
        vector<float> self_volumes(gtree->total_tree_size);
        vector<float> volumes(gtree->total_tree_size);
        vector<float> energies(gtree->total_tree_size);
        vector<float> gammas(gtree->total_tree_size);
        vector<int> last_atom(gtree->total_tree_size);
        vector<int> level(gtree->total_tree_size);
        vector<int> parent(gtree->total_tree_size);
        vector<int> children_start_index(gtree->total_tree_size);
        vector<int> children_count(gtree->total_tree_size);
        vector<int> children_reported(gtree->total_tree_size);
        vector<float4> g(gtree->total_tree_size);
        vector<float4> dv2(gtree->total_tree_size);
        vector<float4> dv1(gtree->total_tree_size);
        vector<float> sfp(gtree->total_tree_size);
        vector<int> size(gtree->num_sections);
        vector<int> tree_pointer_t(gtree->num_sections);
        vector<int> processed(gtree->total_tree_size);
        vector<int> oktoprocess(gtree->total_tree_size);


        gtree->ovSelfVolume->download(self_volumes);
        gtree->ovVolume->download(volumes);
        gtree->ovVolEnergy->download(energies);
        gtree->ovLevel->download(level);
        gtree->ovLastAtom->download(last_atom);
        gtree->ovRootIndex->download(parent);
        gtree->ovChildrenStartIndex->download(children_start_index);
        gtree->ovChildrenCount->download(children_count);
        gtree->ovChildrenReported->download(children_reported);
        gtree->ovG->download(g);
        gtree->ovGamma1i->download(gammas);
        gtree->ovDV1->download(dv1);
        gtree->ovDV2->download(dv2);
        gtree->ovVSfp->download(sfp);
        gtree->ovAtomTreeSize->download(size);
        gtree->ovTreePointer->download(tree_pointer_t);
        gtree->ovProcessedFlag->download(processed);
        gtree->ovOKtoProcessFlag->download(oktoprocess);


        std::cout << "Tree:" << std::endl;
        for (int section = 0; section < gtree->num_sections; section++) {
            std::cout << "Tree for sections: " << section << " " << " size= " << size[section] << std::endl;
            int pp = tree_pointer_t[section];
            int np = gtree->padded_tree_size[section];
            //self_volume += self_volumes[pp];
            std::cout
                    << "slot level LastAtom parent ChStart ChCount SelfV V gamma Energy a x y z dedx dedy dedz sfp processed ok2process children_reported"
                    << endl;
            for (int i = pp; i < pp + np; i++) {
                int maxprint = pp + 1024;
                if (i < maxprint) {
                    std::cout << std::setprecision(4) << std::setw(6) << i << " " << std::setw(7) << (int) level[i]
                              << " " << std::setw(7) << (int) last_atom[i] << " " << std::setw(7) << (int) parent[i]
                              << " " << std::setw(7) << (int) children_start_index[i] << " " << std::setw(7)
                              << (int) children_count[i] << " " << std::setw(15) << (float) self_volumes[i] << " "
                              << std::setw(10) << (float) volumes[i] << " " << std::setw(10) << (float) gammas[i] << " "
                              << std::setw(10) << (float) energies[i] << " " << std::setw(10) << g[i].w << " "
                              << std::setw(10) << g[i].x << " " << std::setw(10) << g[i].y << " " << std::setw(10)
                              << g[i].z << " " << std::setw(10) << dv2[i].x << " " << std::setw(10) << dv2[i].y << " "
                              << std::setw(10) << dv2[i].z << " " << std::setw(10) << sfp[i] << " " << processed[i]
                              << " " << oktoprocess[i] << " " << children_reported[i] << std::endl;
                }
            }
        }
        //std::cout << "Volume (from self volumes):" << self_volume <<std::endl;
    }

    double volume2;
    if (verbose_level > 1) {
        //print self volumes
        vector<float> self_volumes(cu.getPaddedNumAtoms());
        selfVolume->download(self_volumes);
        for (int i = 0; i < numParticles; i++) {
            printf("self_volume: %6.6f atom: %d\n", self_volumes[i], i);
            volume2+=self_volumes[i];
        }
    }

    gtree->ovAtomTreePointer->download(atom_pointer);
    gtree->ovVolEnergy->download(vol_energies);
    energy = 0;
    for (int i = 0; i < numParticles; i++) {
        int slot = atom_pointer[i];
        energy += vol_energies[slot];
        printf("vol_energies[%d]: %6.6f\n", slot, vol_energies[slot]);
    }
    //cout << "Volume 2: " << volume2/ANG3 << endl;
    //cout << "Volume Energy 2:" << energy << endl << endl;


    if (verbose_level > 1) cout << "Done with GVolSA" << endl;

    return 0.0;
}

void GKNPPlugin::CudaCalcGKNPForceKernel::copyParametersToContext(ContextImpl &context, const GKNPForce &force) {
    if (force.getNumParticles() != numParticles) {
        cout << force.getNumParticles() << " != " << numParticles << endl; //Debug
        throw OpenMMException("copyParametersToContext: GKNP plugin does not support changing the number of atoms.");
    }
    if (numParticles == 0)
        return;
    for (int i = 0; i < numParticles; i++) {
        double radius, gamma, alpha, charge;
        bool ishydrogen;
        force.getParticleParameters(i, radius, gamma, alpha, charge, ishydrogen);
        if (pow(radiusVector2[i] - radius, 2) > 1.e-6) {
            throw OpenMMException("updateParametersInContext: GKNP plugin does not support changing atomic radii.");
        }
        int h = ishydrogen ? 1 : 0;
        if (ishydrogenVector[i] != h) {
            throw OpenMMException(
                    "updateParametersInContext: GKNP plugin does not support changing heavy/hydrogen atoms.");
        }
        double g = ishydrogen ? 0 : gamma / roffset;
        gammaVector1[i] = (float) g;
        gammaVector2[i] = (float) -g;
        alphaVector[i] = (float) alpha;
        chargeVector[i] = (float) charge;
    }
    gammaParam1->upload(gammaVector1);
    gammaParam2->upload(gammaVector2);
    alphaParam->upload(alphaVector);
    chargeParam->upload(chargeVector);
}

