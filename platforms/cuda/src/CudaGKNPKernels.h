#ifndef CUDA_GKNP_KERNELS_H_
#define CUDA_GKNP_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                           OpenMM-GKNP                                    *
 * -------------------------------------------------------------------------- */

#include "GKNPKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/reference/RealVec.h"
using namespace std;

namespace GKNPPlugin {

/**
 * This kernel is invoked by GKNPForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcGKNPForceKernel : public CalcGKNPForceKernel {
public:
    CudaCalcGKNPForceKernel(std::string name, const OpenMM::Platform &platform, OpenMM::CudaContext &cu,
                            const OpenMM::System &system) :
            CalcGKNPForceKernel(name, platform), cu(cu), system(system) {

        hasCreatedKernels = false;
        hasInitializedKernels = false;

        radtypeScreened = NULL;
        radtypeScreener = NULL;

        selfVolume = NULL;
        selfVolumeLargeR = NULL;
        Semaphor = NULL;
        volScalingFactor = NULL;
        BornRadius = NULL;
        invBornRadius = NULL;
        invBornRadius_fp = NULL;
        GBDerY = NULL;
        GBDerBrU = NULL;
        GBDerU = NULL;
        VdWDerBrW = NULL;
        VdWDerW = NULL;

        GaussianExponent = NULL;
        GaussianVolume = NULL;
        GaussianExponentLargeR = NULL;
        GaussianVolumeLargeR = NULL;

        AtomicGamma = NULL;
        grad = NULL;



        //i4_lut = NULL;

        PanicButton = NULL;
        pinnedPanicButtonBuffer = NULL;
    }

    ~CudaCalcGKNPForceKernel();

    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the GKNPForce this kernel will be used for
     */
    void initialize(const OpenMM::System &system, const GKNPForce &force);

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl &context, bool includeForces, bool includeEnergy);

    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GKNPForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl &context, const GKNPForce &force);

    class CudaOverlapTree {
    public:
        CudaOverlapTree(void) {
            ovAtomTreePointer = NULL;
            ovAtomTreeSize = NULL;
            ovTreePointer = NULL;
            ovNumAtomsInTree = NULL;
            ovFirstAtom = NULL;
            NIterations = NULL;
            ovAtomTreePaddedSize = NULL;
            ovAtomTreeLock = NULL;
            ovLevel = NULL;
            ovG = NULL;
            ovVolume = NULL;
            ovVsp = NULL;
            ovVSfp = NULL;
            ovSelfVolume = NULL;
            ovVolEnergy = NULL;
            ovGamma1i = NULL;
            ovDV1 = NULL;
            ovDV2 = NULL;
            ovPF = NULL;
            ovLastAtom = NULL;
            ovRootIndex = NULL;
            ovChildrenStartIndex = NULL;
            ovChildrenCount = NULL;
            ovChildrenCountTop = NULL;
            ovChildrenCountBottom = NULL;
            ovProcessedFlag = NULL;
            ovOKtoProcessFlag = NULL;
            ovChildrenReported = NULL;

            ovAtomBuffer = NULL;
            selfVolumeBuffer_long = NULL;
            selfVolumeBuffer = NULL;
            AccumulationBuffer1_long = NULL;
            AccumulationBuffer1_real = NULL;
            AccumulationBuffer2_long = NULL;
            AccumulationBuffer2_real = NULL;
            gradBuffers_long = NULL;

            temp_buffer_size = -1;
            gvol_buffer_temp = NULL;
            tree_pos_buffer_temp = NULL;
            i_buffer_temp = NULL;
            atomj_buffer_temp = NULL;

            has_saved_noverlaps = false;
            tree_size_boost = 2;//6;//debug 2 is default

            hasExceededTempBuffer = false;

        };

        ~CudaOverlapTree(void) {
            delete ovAtomTreePointer;
            delete ovAtomTreeSize;
            delete ovTreePointer;
            delete ovNumAtomsInTree;
            delete ovFirstAtom;
            delete NIterations;
            delete ovAtomTreePaddedSize;
            delete ovAtomTreeLock;
            delete ovLevel;
            delete ovG;
            delete ovVolume;
            delete ovVsp;
            delete ovVSfp;
            delete ovSelfVolume;
            delete ovVolEnergy;
            delete ovGamma1i;
            delete ovDV1;
            delete ovDV2;
            delete ovPF;
            delete ovLastAtom;
            delete ovRootIndex;
            delete ovChildrenStartIndex;
            delete ovChildrenCount;
            delete ovChildrenCountTop;
            delete ovChildrenCountBottom;
            delete ovProcessedFlag;
            delete ovOKtoProcessFlag;
            delete ovChildrenReported;

            delete ovAtomBuffer;
            delete selfVolumeBuffer_long;
            delete selfVolumeBuffer;
            delete AccumulationBuffer1_long;
            delete AccumulationBuffer1_real;
            delete AccumulationBuffer2_long;
            delete AccumulationBuffer2_real;
            delete gradBuffers_long;

            delete gvol_buffer_temp;
            delete tree_pos_buffer_temp;
            delete i_buffer_temp;
            delete atomj_buffer_temp;
        };

        //initializes tree sections and sizes with number of atoms and number of overlaps
        void init_tree_size(int num_atoms, int padded_num_atoms, int num_compute_units, int pad_modulo,
                            vector<int> &noverlaps_current);

        //simpler version with precomputed tree sizes
        void init_tree_size(int padded_num_atoms, int tree_section_size, int num_compute_units, int pad_modulo);

        //resizes tree buffers
        void resize_tree_buffers(OpenMM::CudaContext &cu, int ov_work_group_size);

        //copies the tree framework to Cuda device memory
        int copy_tree_to_device(void);


        // host variables and buffers
        int num_atoms;
        int padded_num_atoms;
        int total_atoms_in_tree;
        int total_tree_size;
        int num_sections;
        vector<int> tree_size;
        vector<int> padded_tree_size;
        vector<int> atom_tree_pointer; //pointers to 1-body atom slots
        vector<int> tree_pointer;      //pointers to tree sections
        vector<int> natoms_in_tree;    //no. atoms in each tree section
        vector<int> first_atom;        //the first atom in each tree section

        /* overlap tree buffers on Device */
        OpenMM::CudaArray *ovAtomTreePointer;
        OpenMM::CudaArray *ovAtomTreeSize;
        OpenMM::CudaArray *ovTreePointer;
        OpenMM::CudaArray *ovNumAtomsInTree;
        OpenMM::CudaArray *ovFirstAtom;
        OpenMM::CudaArray *NIterations;
        OpenMM::CudaArray *ovAtomTreePaddedSize;
        OpenMM::CudaArray *ovAtomTreeLock;
        OpenMM::CudaArray *ovLevel;
        OpenMM::CudaArray *ovG; // real4: Gaussian position + exponent
        OpenMM::CudaArray *ovVolume;
        OpenMM::CudaArray *ovVsp;
        OpenMM::CudaArray *ovVSfp;
        OpenMM::CudaArray *ovSelfVolume;
        OpenMM::CudaArray *ovVolEnergy;
        OpenMM::CudaArray *ovGamma1i;
        /* volume derivatives */
        OpenMM::CudaArray *ovDV1; // real4: dV12/dr1 + dV12/dV1 for each overlap
        OpenMM::CudaArray *ovDV2; // volume gradient accumulator
        OpenMM::CudaArray *ovPF;  //(P) and (F) aux variables

        OpenMM::CudaArray *ovLastAtom;
        OpenMM::CudaArray *ovRootIndex;
        OpenMM::CudaArray *ovChildrenStartIndex;
        OpenMM::CudaArray *ovChildrenCount;
        OpenMM::CudaArray *ovChildrenCountTop;
        OpenMM::CudaArray *ovChildrenCountBottom;
        OpenMM::CudaArray *ovProcessedFlag;
        OpenMM::CudaArray *ovOKtoProcessFlag;
        OpenMM::CudaArray *ovChildrenReported;

        OpenMM::CudaArray *ovAtomBuffer;
        OpenMM::CudaArray *selfVolumeBuffer_long;
        OpenMM::CudaArray *selfVolumeBuffer;
        OpenMM::CudaArray *AccumulationBuffer1_long;
        OpenMM::CudaArray *AccumulationBuffer1_real;
        OpenMM::CudaArray *AccumulationBuffer2_long;
        OpenMM::CudaArray *AccumulationBuffer2_real;
        OpenMM::CudaArray *gradBuffers_long;

        int temp_buffer_size;
        OpenMM::CudaArray *gvol_buffer_temp;
        OpenMM::CudaArray *tree_pos_buffer_temp;
        OpenMM::CudaArray *i_buffer_temp;
        OpenMM::CudaArray *atomj_buffer_temp;

        double tree_size_boost;
        int has_saved_noverlaps;
        vector<int> saved_noverlaps;

        bool hasExceededTempBuffer;
    };//class CudaOverlapTree


private:
    const GKNPForce *gvol_force;

    int numParticles;
    unsigned int version;
    bool useCutoff;
    bool usePeriodic;
    bool useExclusions;
    double cutoffDistance;
    double roffset;
    float common_gamma;
    int maxTiles;
    bool hasInitializedKernels;
    bool hasCreatedKernels;
    OpenMM::CudaContext &cu;
    const OpenMM::System &system;
    int ov_work_group_size; //thread group size
    int num_compute_units;

    CudaOverlapTree *gtree;   //tree of atomic overlaps

    double solvent_radius; //solvent probe radius for GKNP2

    OpenMM::CudaArray *radiusParam1;
    OpenMM::CudaArray *radiusParam2;
    OpenMM::CudaArray *gammaParam1;
    OpenMM::CudaArray *gammaParam2;
    OpenMM::CudaArray *ishydrogenParam;
    OpenMM::CudaArray *chargeParam;
    OpenMM::CudaArray *alphaParam;

    //C++ vectors corresponding to parameter buffers above
    vector<float> radiusVector1; //enlarged radii
    vector<float> radiusVector2; //vdw radii
    vector<float> gammaVector1;  //gamma/radius_offset
    vector<float> gammaVector2;  //-gamma/radius_offset
    vector<float> chargeVector;  //charge
    vector<float> alphaVector;   //alpha vdw parameter
    vector<int> ishydrogenVector;

    OpenMM::CudaArray *testBuffer;

    OpenMM::CudaArray *radtypeScreened;
    OpenMM::CudaArray *radtypeScreener;

    OpenMM::CudaArray *selfVolume; //vdw radii
    OpenMM::CudaArray *selfVolumeLargeR; //large radii
    OpenMM::CudaArray *Semaphor;
    OpenMM::CudaArray *volScalingFactor;
    OpenMM::CudaArray *BornRadius;
    OpenMM::CudaArray *invBornRadius;
    OpenMM::CudaArray *invBornRadius_fp;
    OpenMM::CudaArray *GBDerY;
    OpenMM::CudaArray *GBDerBrU;
    OpenMM::CudaArray *GBDerU;
    OpenMM::CudaArray *VdWDerBrW;
    OpenMM::CudaArray *VdWDerW;
    OpenMM::CudaArray *grad;

    CUfunction resetBufferKernel;
    CUfunction resetOvCountKernel;
    CUfunction resetTree;
    CUfunction resetSelfVolumesKernel;
    CUfunction InitOverlapTreeKernel_1body_1;
    CUfunction InitOverlapTreeKernel_1body_2;

    CUfunction InitOverlapTreeCountKernel;

    CUfunction reduceovCountBufferKernel;

    CUfunction InitOverlapTreeKernel;
    int InitOverlapTreeKernel_first_nbarg;

    CUfunction ComputeOverlapTreeKernel;
    CUfunction ComputeOverlapTree_1passKernel;
    CUfunction computeSelfVolumesKernel;
    CUfunction reduceSelfVolumesKernel_tree;
    CUfunction reduceSelfVolumesKernel_buffer;
    CUfunction updateSelfVolumesForcesKernel;

    CUfunction resetTreeKernel;
    CUfunction SortOverlapTree2bodyKernel;
    CUfunction resetComputeOverlapTreeKernel;
    CUfunction ResetRescanOverlapTreeKernel;
    CUfunction InitRescanOverlapTreeKernel;
    CUfunction RescanOverlapTreeKernel;
    CUfunction RescanOverlapTreeGammasKernel_W;
    CUfunction InitOverlapTreeGammasKernel_1body_W;
    //CUfunction computeVolumeEnergyKernel;

    /* Gaussian atomic parameters */
    vector<float> gaussian_exponent;
    vector<float> gaussian_volume;
    OpenMM::CudaArray *GaussianExponent;
    OpenMM::CudaArray *GaussianVolume;
    OpenMM::CudaArray *GaussianExponentLargeR;
    OpenMM::CudaArray *GaussianVolumeLargeR;

    /* gamma parameters */
    vector<float> atomic_gamma;
    OpenMM::CudaArray *AtomicGamma;
    vector<int> atom_ishydrogen;

    int niterations;
    int verbose_level;

    void executeInitKernels(OpenMM::ContextImpl &context, bool includeForces, bool includeEnergy);

    double executeGVolSA(OpenMM::ContextImpl &context, bool includeForces, bool includeEnergy);

    //TODO: Panic Button?
    //flag to give up
    OpenMM::CudaArray *PanicButton;
    vector<int> panic_button;
    int *pinnedPanicButtonBuffer;
    CUevent downloadPanicButtonEvent;
};

 
} // namespace GKNPPlugin

#endif /*CUDA_GKNP_KERNELS_H_*/
