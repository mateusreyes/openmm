/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2013-2025 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors: Evan Pretti                                                  *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#ifdef _MSC_VER
  #define POCKETFFT_NO_VECTORS
#endif
#define POCKETFFT_CACHE_SIZE 4
#include "CpuPmeKernels.h"
#include "SimTKOpenMMRealType.h"
#include "ReferenceForce.h"
#include "openmm/internal/hardware.h"
#include "openmm/internal/vectorize.h"
#include "openmm/OpenMMException.h"
#include "pocketfft_hdronly.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <cstdlib>

using namespace OpenMM;
using namespace std;

static const int PME_ORDER = 5;

bool CpuCalcDispersionPmeReciprocalForceKernel::hasInitializedThreads = false;
int CpuCalcDispersionPmeReciprocalForceKernel::numThreads = 0;

// X-slab partitioned spreadCharge.
//
// Each thread owns a contiguous X-range [my_x_start, my_x_end) of the grid and
// writes only to that slice. The caller supplies the precomputed list of
// particles whose B-spline X-stencil overlaps this thread's X-range
// (my_particles). The grid is shared across all threads; because X-slices are
// disjoint there is no false sharing and no reduction step is needed.
//
// This thread is responsible for zeroing its own X-slice of the shared grid.
static void spreadCharge(float* posq, vector<float>& grid, int gridx, int gridy, int gridz,
        Vec3* periodicBoxVectors, Vec3* recipBoxVectors,
        const float epsilonFactor, int my_x_start, int my_x_end, const vector<int>& my_particles) {
    float temp[4];
    fvec4 boxSize((float) periodicBoxVectors[0][0], (float) periodicBoxVectors[1][1], (float) periodicBoxVectors[2][2], 0);
    fvec4 invBoxSize((float) recipBoxVectors[0][0], (float) recipBoxVectors[1][1], (float) recipBoxVectors[2][2], 0);
    fvec4 recipBoxVec0((float) recipBoxVectors[0][0], (float) recipBoxVectors[0][1], (float) recipBoxVectors[0][2], 0);
    fvec4 recipBoxVec1((float) recipBoxVectors[1][0], (float) recipBoxVectors[1][1], (float) recipBoxVectors[1][2], 0);
    fvec4 recipBoxVec2((float) recipBoxVectors[2][0], (float) recipBoxVectors[2][1], (float) recipBoxVectors[2][2], 0);
    fvec4 gridSize(gridx, gridy, gridz, 0);
    ivec4 gridSizeInt(gridx, gridy, gridz, 0);
    fvec4 one(1);
    fvec4 scale(1.0f/(PME_ORDER-1));
    float posInBox[4] = {0,0,0,0};

    // Zero only this thread's X-slice of the shared grid.
    const size_t sliceLen = (size_t)(my_x_end - my_x_start) * gridy * gridz;
    if (sliceLen > 0)
        memset(&grid[(size_t)my_x_start * gridy * gridz], 0, sliceLen * sizeof(float));

    const int numMine = (int) my_particles.size();
    for (int idx = 0; idx < numMine; ++idx) {
        int i = my_particles[idx];
        // Find the position relative to the nearest grid point.

        fvec4 pos(&posq[4*i]);
        (pos-boxSize*floor(pos*invBoxSize)).store(posInBox);
        fvec4 t = posInBox[0]*recipBoxVec0 + posInBox[1]*recipBoxVec1 + posInBox[2]*recipBoxVec2;
        t = (t-floor(t))*gridSize;
        ivec4 ti = t;
        fvec4 dr = t-ti;
        ivec4 gridIndex = ti-(gridSizeInt&ti==gridSizeInt);

        // Compute the B-spline coefficients.

        fvec4 data[PME_ORDER];
        data[PME_ORDER-1] = 0.0f;
        data[1] = dr;
        data[0] = one-dr;
        for (int j = 3; j < PME_ORDER; j++) {
            fvec4 div(1.0f/(j-1));
            data[j-1] = div*dr*data[j-2];
            for (int k = 1; k < j-1; k++)
                data[j-k-1] = div*((dr+k)*data[j-k-2]+(fvec4(j-k)-dr)*data[j-k-1]);
            data[0] = div*(one-dr)*data[0];
        }
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < (PME_ORDER-1); j++)
            data[PME_ORDER-j-1] = scale*((dr+j)*data[PME_ORDER-j-2]+(fvec4(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(one-dr)*data[0];

        // Spread the charges.

        int gridIndexX = gridIndex[0];
        int gridIndexY = gridIndex[1];
        int gridIndexZ = gridIndex[2];
        if (gridIndexX < 0)
            return; // This happens when a simulation blows up and coordinates become NaN.
        int zindex[PME_ORDER];
        for (int j = 0; j < PME_ORDER; j++) {
            zindex[j] = gridIndexZ+j;
            zindex[j] -= (zindex[j] >= gridz ? gridz : 0);
        }
        float charge = epsilonFactor*posq[4*i+3];
        fvec4 zdata0to3(data[0][2], data[1][2], data[2][2], data[3][2]);
        float zdata4 = data[4][2];
        if (gridIndexZ+4 < gridz) {
            for (int ix = 0; ix < PME_ORDER; ix++) {
                int x = gridIndexX+ix;
                x -= (x >= gridx ? gridx : 0);
                // Skip stencil cells that fall outside this thread's X-slice.
                if (x < my_x_start || x >= my_x_end) continue;
                int xbase = x*gridy*gridz;
                float xdata = charge*data[ix][0];
                for (int iy = 0; iy < PME_ORDER; iy++) {
                    int ybase = gridIndexY+iy;
                    ybase -= (ybase >= gridy ? gridy : 0);
                    ybase = xbase + ybase*gridz;
                    float multiplier = xdata*data[iy][1];
                    fvec4 add0to3 = zdata0to3*multiplier;
                    (fvec4(&grid[ybase+gridIndexZ])+add0to3).store(&grid[ybase+gridIndexZ]);
                    grid[ybase+zindex[4]] += multiplier*zdata4;
                }
            }
        }
        else {
            for (int ix = 0; ix < PME_ORDER; ix++) {
                int x = gridIndexX+ix;
                x -= (x >= gridx ? gridx : 0);
                if (x < my_x_start || x >= my_x_end) continue;
                int xbase = x*gridy*gridz;
                float xdata = charge*data[ix][0];
                for (int iy = 0; iy < PME_ORDER; iy++) {
                    int ybase = gridIndexY+iy;
                    ybase -= (ybase >= gridy ? gridy : 0);
                    ybase = xbase + ybase*gridz;
                    float multiplier = xdata*data[iy][1];
                    fvec4 add0to3 = zdata0to3*multiplier;
                    add0to3.store(temp);
                    grid[ybase+zindex[0]] += temp[0];
                    grid[ybase+zindex[1]] += temp[1];
                    grid[ybase+zindex[2]] += temp[2];
                    grid[ybase+zindex[3]] += temp[3];
                    grid[ybase+zindex[4]] += multiplier*zdata4;
                }
            }
        }
    }
}

// Build per-thread particle lists for X-slab partitioning. For every particle,
// determine which thread's X-range its B-spline X-stencil overlaps (including
// the periodic wrap-around case) and append the particle to each owner's list.
//
// This runs in a single thread (thread 0) to keep the implementation simple and
// to preserve deterministic ordering of writes per X-owner.
static void buildThreadParticleLists(float* posq, int numParticles, int gridx, int gridy, int gridz,
        int numThreads, Vec3* periodicBoxVectors, Vec3* recipBoxVectors,
        vector<vector<int> >& threadParticles) {
    fvec4 boxSize((float) periodicBoxVectors[0][0], (float) periodicBoxVectors[1][1], (float) periodicBoxVectors[2][2], 0);
    fvec4 invBoxSize((float) recipBoxVectors[0][0], (float) recipBoxVectors[1][1], (float) recipBoxVectors[2][2], 0);
    fvec4 recipBoxVec0((float) recipBoxVectors[0][0], (float) recipBoxVectors[0][1], (float) recipBoxVectors[0][2], 0);
    fvec4 recipBoxVec1((float) recipBoxVectors[1][0], (float) recipBoxVectors[1][1], (float) recipBoxVectors[1][2], 0);
    fvec4 recipBoxVec2((float) recipBoxVectors[2][0], (float) recipBoxVectors[2][1], (float) recipBoxVectors[2][2], 0);
    fvec4 gridSize(gridx, gridy, gridz, 0);
    ivec4 gridSizeInt(gridx, gridy, gridz, 0);

    for (int t = 0; t < numThreads; t++)
        threadParticles[t].clear();

    // Precompute each thread's X-range so we can map x -> owner thread quickly.
    // x_to_thread[x] = the thread that owns grid X-coordinate x.
    vector<int> x_to_thread(gridx);
    for (int t = 0; t < numThreads; t++) {
        int xStart = (t*gridx)/numThreads;
        int xEnd = ((t+1)*gridx)/numThreads;
        for (int x = xStart; x < xEnd; x++)
            x_to_thread[x] = t;
    }

    float posInBox[4] = {0,0,0,0};
    // Reserve a rough estimate: each particle touches ~5 X-cells, expected to
    // be in 1-2 threads. Reserve 1.5x numParticles/numThreads per list.
    const int reserveHint = max(16, (3 * numParticles) / (2 * numThreads));
    for (int t = 0; t < numThreads; t++)
        threadParticles[t].reserve(reserveHint);

    for (int i = 0; i < numParticles; ++i) {
        fvec4 pos(&posq[4*i]);
        (pos-boxSize*floor(pos*invBoxSize)).store(posInBox);
        fvec4 tv = posInBox[0]*recipBoxVec0 + posInBox[1]*recipBoxVec1 + posInBox[2]*recipBoxVec2;
        tv = (tv-floor(tv))*gridSize;
        ivec4 ti = tv;
        ivec4 gridIndex = ti-(gridSizeInt&ti==gridSizeInt);
        int gridIndexX = gridIndex[0];
        if (gridIndexX < 0) {
            // NaN/blown-up coords. spreadCharge would early-return; skip binning.
            continue;
        }

        // The X-stencil is [gridIndexX, gridIndexX+PME_ORDER), wrapping at gridx.
        // Find the set of threads whose X-ranges overlap this stencil and add i
        // to each one's list. Particles typically span 1-2 X-ranges so this is
        // cheap. We track owners we have already added for this particle in a
        // tiny fixed-size buffer (at most PME_ORDER=5 unique owners).
        int seenOwners[PME_ORDER];
        int numSeen = 0;
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int x = gridIndexX + ix;
            x -= (x >= gridx ? gridx : 0);
            int owner = x_to_thread[x];
            bool already = false;
            for (int s = 0; s < numSeen; s++) {
                if (seenOwners[s] == owner) { already = true; break; }
            }
            if (!already) {
                seenOwners[numSeen++] = owner;
                threadParticles[owner].push_back(i);
            }
        }
    }
}

#define FAST_ERFC 1
static void computeReciprocalDispersionEterm(int start, int end, int gridx, int gridy, int gridz, vector<float>& recipEterm, double alpha, vector<float>* bsplineModuli, Vec3* periodicBoxVectors, Vec3* recipBoxVectors) {
    const unsigned int zsize = gridz/2+1;
    const unsigned int yzsize = gridy*zsize;
    const float scaleFactor = (float)  -2.0f*M_PI*sqrtf(M_PI) / (6.0*periodicBoxVectors[0][0]*periodicBoxVectors[1][1]*periodicBoxVectors[2][2]);

    float bfac = M_PI / alpha;
    float fac1 = 2.0f*M_PI*M_PI*M_PI*sqrtf(M_PI);
    float fac2 = alpha*alpha*alpha;
    float fac3 = -2.0f*alpha*M_PI*M_PI;
    float b, m, m3, expterm, erfcterm, t;

    for (int kx = start; kx < end; kx++) {
        int mx = (kx < (gridx+1)/2) ? kx : kx-gridx;
        float mhx = mx*(float)recipBoxVectors[0][0];
        float bx = bsplineModuli[0][kx];
        for (int ky = 0; ky < gridy; ky++) {
            int my = (ky < (gridy+1)/2) ? ky : ky-gridy;
            float mhy = mx*(float)recipBoxVectors[1][0] + my*(float)recipBoxVectors[1][1];
            float mhx2y2 = mhx*mhx + mhy*mhy;
            float bxby = bx*bsplineModuli[1][ky];
            for (int kz = 0; kz < zsize; kz++) {
                int index = kx*yzsize + ky*zsize + kz;
                int mz = (kz < (gridz+1)/2) ? kz : kz-gridz;
                float mhz = mx*(float)recipBoxVectors[2][0] + my*(float)recipBoxVectors[2][1] + mz*(float)recipBoxVectors[2][2];
                float bz = bsplineModuli[2][kz];
                float m2 = mhx2y2 + mhz*mhz;
                float denom = scaleFactor/(bxby*bz);

                m = sqrtf(m2);
                m3 = m*m2;
                b = bfac*m;
                expterm = exp(-b*b);

#if FAST_ERFC
                // This approximation for erfc is from Abramowitz and Stegun (1964) p. 299.  They cite the following as
                // the original source: C. Hastings, Jr., Approximations for Digital Computers (1955).  It has a maximum
                // error of 1.5e-7.  Stolen by ACS from the CUDA platform's AMOEBA plugin.
                t = 1.0f/(1.0f+0.3275911f*b);
                erfcterm = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expterm;
#else
                erfcterm = erfc(b);
#endif
                recipEterm[index] = (fac1*erfcterm*m3 + expterm*(fac2 + fac3*m2)) * denom;
            }
        }
    }
}

static void computeReciprocalEterm(int start, int end, int gridx, int gridy, int gridz, vector<float>& recipEterm, double alpha, vector<float>* bsplineModuli, Vec3* periodicBoxVectors, Vec3* recipBoxVectors) {
    const unsigned int zsize = gridz/2+1;
    const unsigned int yzsize = gridy*zsize;
    const float scaleFactor = (float) (M_PI*periodicBoxVectors[0][0]*periodicBoxVectors[1][1]*periodicBoxVectors[2][2]);
    const float recipExpFactor = (float) (M_PI*M_PI/(alpha*alpha));

    int firstz = (start == 0 ? 1 : 0);
    for (int kx = start; kx < end; kx++) {
        int mx = (kx < (gridx+1)/2) ? kx : kx-gridx;
        float mhx = mx*(float)recipBoxVectors[0][0];
        float bx = scaleFactor*bsplineModuli[0][kx];
        for (int ky = 0; ky < gridy; ky++) {
            int my = (ky < (gridy+1)/2) ? ky : ky-gridy;
            float mhy = mx*(float)recipBoxVectors[1][0] + my*(float)recipBoxVectors[1][1];
            float mhx2y2 = mhx*mhx + mhy*mhy;
            float bxby = bx*bsplineModuli[1][ky];
            for (int kz = firstz; kz < zsize; kz++) {
                int index = kx*yzsize + ky*zsize + kz;
                int mz = (kz < (gridz+1)/2) ? kz : kz-gridz;
                float mhz = mx*(float)recipBoxVectors[2][0] + my*(float)recipBoxVectors[2][1] + mz*(float)recipBoxVectors[2][2];
                float bz = bsplineModuli[2][kz];
                float m2 = mhx2y2 + mhz*mhz;
                float denom = m2*bxby*bz;
                recipEterm[index] = exp(-recipExpFactor*m2)/denom;
            }
            firstz = 0;
        }
    }
}

static double reciprocalEnergy(int start, int end, vector<complex<float> >& grid, vector<float>& recipEterm, int gridx, int gridy, int gridz, double alpha, vector<float>* bsplineModuli, Vec3* periodicBoxVectors, Vec3* recipBoxVectors) {
    const unsigned int zsizeHalf = gridz/2+1;
    const unsigned int yzsizeHalf = gridy*zsizeHalf;

    double energy = 0.0;

    int firstz = (start == 0 ? 1 : 0);
    for (int kx = start; kx < end; kx++) {
        for (int ky = 0; ky < gridy; ky++) {
            for (int kz = firstz; kz < gridz; kz++) {
                int kx1, ky1, kz1;
                if (kz >= gridz/2+1) {
                    kx1 = (kx == 0 ? kx : gridx-kx);
                    ky1 = (ky == 0 ? ky : gridy-ky);
                    kz1 = gridz-kz;
                }
                else {
                    kx1 = kx;
                    ky1 = ky;
                    kz1 = kz;
                }
                int index = kx1*yzsizeHalf + ky1*zsizeHalf + kz1;
                float gridReal = grid[index].real();
                float gridImag = grid[index].imag();
                energy += recipEterm[index]*(gridReal*gridReal+gridImag*gridImag);
            }
            firstz = 0;
        }
    }
    return 0.5*energy;
}


static double reciprocalDispersionEnergy(int start, int end, vector<complex<float> >& grid, const vector<float>& recipEterm, int gridx, int gridy, int gridz, double alpha, vector<float>* bsplineModuli, Vec3* periodicBoxVectors, Vec3* recipBoxVectors) {
    const unsigned int zsizeHalf = gridz/2+1;
    const unsigned int yzsizeHalf = gridy*zsizeHalf;

    double energy = 0.0;

    for (int kx = start; kx < end; kx++) {
        for (int ky = 0; ky < gridy; ky++) {
            for (int kz = 0; kz < gridz; kz++) {
                int kx1, ky1, kz1;
                if (kz >= gridz/2+1) {
                    kx1 = (kx == 0 ? kx : gridx-kx);
                    ky1 = (ky == 0 ? ky : gridy-ky);
                    kz1 = gridz-kz;
                }
                else {
                    kx1 = kx;
                    ky1 = ky;
                    kz1 = kz;
                }
                int index = kx1*yzsizeHalf + ky1*zsizeHalf + kz1;
                float gridReal = grid[index].real();
                float gridImag = grid[index].imag();
                energy += recipEterm[index]*(gridReal*gridReal+gridImag*gridImag);
            }
        }
    }
    return 0.5f*energy;
}


static void reciprocalConvolution(int start, int end, vector<complex<float> >& grid, vector<float>& recipEterm) {
    for (int index = start; index < end; index++) {
        float eterm = recipEterm[index];
        grid[index] *= eterm;
    }
}

static void interpolateForces(float* posq, vector<float>& force, vector<float>& grid, int gridx, int gridy, int gridz, int numParticles, Vec3* periodicBoxVectors, Vec3* recipBoxVectors, atomic<int>& atomicCounter, const float epsilonFactor, int numThreads) {
    fvec4 boxSize((float) periodicBoxVectors[0][0], (float) periodicBoxVectors[1][1], (float) periodicBoxVectors[2][2], 0);
    fvec4 invBoxSize((float) recipBoxVectors[0][0], (float) recipBoxVectors[1][1], (float) recipBoxVectors[2][2], 0);
    fvec4 recipBoxVec0((float) recipBoxVectors[0][0], (float) recipBoxVectors[0][1], (float) recipBoxVectors[0][2], 0);
    fvec4 recipBoxVec1((float) recipBoxVectors[1][0], (float) recipBoxVectors[1][1], (float) recipBoxVectors[1][2], 0);
    fvec4 recipBoxVec2((float) recipBoxVectors[2][0], (float) recipBoxVectors[2][1], (float) recipBoxVectors[2][2], 0);
    fvec4 gridSize(gridx, gridy, gridz, 0);
    ivec4 gridSizeInt(gridx, gridy, gridz, 0);
    fvec4 one(1);
    fvec4 scale(1.0f/(PME_ORDER-1));

    const int groupSize = max(1, numParticles / (10 * numThreads));
    while (true) {
        int start = atomicCounter.fetch_add(groupSize);
        if (start >= numParticles)
            break;

        int end = min(start + groupSize, numParticles);

        for (int i = start; i < end; i++) {
            // Find the position relative to the nearest grid point.

            fvec4 pos(&posq[4*i]);
            float posInBox[4];
            (pos-boxSize*floor(pos*invBoxSize)).store(posInBox);
            fvec4 t = posInBox[0]*recipBoxVec0 + posInBox[1]*recipBoxVec1 + posInBox[2]*recipBoxVec2;
            t = (t-floor(t))*gridSize;
            ivec4 ti = t;
            fvec4 dr = t-ti;
            ivec4 gridIndex = ti-(gridSizeInt&ti==gridSizeInt);

            // Compute the B-spline coefficients.

            fvec4 data[PME_ORDER];
            fvec4 ddata[PME_ORDER];
            data[PME_ORDER-1] = 0.0f;
            data[1] = dr;
            data[0] = one-dr;
            for (int j = 3; j < PME_ORDER; j++) {
                fvec4 div(1.0f/(j-1));
                data[j-1] = div*dr*data[j-2];
                for (int k = 1; k < j-1; k++)
                    data[j-k-1] = div*((dr+k)*data[j-k-2]+(fvec4(j-k)-dr)*data[j-k-1]);
                data[0] = div*(one-dr)*data[0];
            }
            ddata[0] = -data[0];
            for (int j = 1; j < PME_ORDER; j++)
                ddata[j] = data[j-1]-data[j];
            data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
            for (int j = 1; j < (PME_ORDER-1); j++)
                data[PME_ORDER-j-1] = scale*((dr+j)*data[PME_ORDER-j-2]+(fvec4(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
            data[0] = scale*(one-dr)*data[0];

            // Compute the force on this atom.

            int gridIndexX = gridIndex[0];
            int gridIndexY = gridIndex[1];
            int gridIndexZ = gridIndex[2];
            if (gridIndexX < 0)
                return; // This happens when a simulation blows up and coordinates become NaN.
            int zindex[PME_ORDER];
            for (int j = 0; j < PME_ORDER; j++) {
                zindex[j] = gridIndexZ+j;
                zindex[j] -= (zindex[j] >= gridz ? gridz : 0);
            }
            fvec4 zdata[PME_ORDER];
            for (int j = 0; j < PME_ORDER; j++)
                zdata[j] = fvec4(data[j][2], data[j][2], ddata[j][2], 0);
            fvec4 f = 0.0f;
            for (int ix = 0; ix < PME_ORDER; ix++) {
                int xbase = gridIndexX+ix;
                xbase -= (xbase >= gridx ? gridx : 0);
                xbase = xbase*gridy*gridz;
                float dx = data[ix][0];
                float ddx = ddata[ix][0];
                fvec4 xdata(ddx, dx, dx, 0);

                for (int iy = 0; iy < PME_ORDER; iy++) {
                    int ybase = gridIndexY+iy;
                    ybase -= (ybase >= gridy ? gridy : 0);
                    ybase = xbase + ybase*gridz;
                    float dy = data[iy][1];
                    float ddy = ddata[iy][1];
                    fvec4 xydata = xdata*fvec4(dy, ddy, dy, 0);

                    for (int iz = 0; iz < PME_ORDER; iz++) {
                        fvec4 gridValue(grid[ybase+zindex[iz]]);
                        f = f+xydata*zdata[iz]*gridValue;
                    }
                }
            }
            f *= -epsilonFactor*posq[4*i+3];
            float fc[4];
            f.store(fc);
            force[4*i+0] = fc[0]*gridx*(float)recipBoxVectors[0][0];
            force[4*i+1] = fc[0]*gridx*(float)recipBoxVectors[1][0]+fc[1]*gridy*(float)recipBoxVectors[1][1];
            force[4*i+2] = fc[0]*gridx*(float)recipBoxVectors[2][0]+fc[1]*gridy*(float)recipBoxVectors[2][1]+fc[2]*gridz*(float)recipBoxVectors[2][2];
        }
    }
}

static void interpolateChargeDerivatives(float* posq, const vector<int>& chargeIndices, vector<float>& chargeDerivatives, vector<float>& grid, int gridx, int gridy, int gridz, int numIndices, Vec3* periodicBoxVectors, Vec3* recipBoxVectors, atomic<int>& atomicCounter, const float epsilonFactor, int numThreads) {
    fvec4 boxSize((float) periodicBoxVectors[0][0], (float) periodicBoxVectors[1][1], (float) periodicBoxVectors[2][2], 0);
    fvec4 invBoxSize((float) recipBoxVectors[0][0], (float) recipBoxVectors[1][1], (float) recipBoxVectors[2][2], 0);
    fvec4 recipBoxVec0((float) recipBoxVectors[0][0], (float) recipBoxVectors[0][1], (float) recipBoxVectors[0][2], 0);
    fvec4 recipBoxVec1((float) recipBoxVectors[1][0], (float) recipBoxVectors[1][1], (float) recipBoxVectors[1][2], 0);
    fvec4 recipBoxVec2((float) recipBoxVectors[2][0], (float) recipBoxVectors[2][1], (float) recipBoxVectors[2][2], 0);
    fvec4 gridSize(gridx, gridy, gridz, 0);
    ivec4 gridSizeInt(gridx, gridy, gridz, 0);
    fvec4 one(1);
    fvec4 scale(1.0f/(PME_ORDER-1));

    const int groupSize = max(1, numIndices / (10 * numThreads));
    while (true) {
        int start = atomicCounter.fetch_add(groupSize);
        if (start >= numIndices)
            break;

        int end = min(start + groupSize, numIndices);

        for (int ii = start; ii < end; ii++) {
            // Find the position relative to the nearest grid point.

            fvec4 pos(&posq[4*chargeIndices[ii]]);
            float posInBox[4];
            (pos-boxSize*floor(pos*invBoxSize)).store(posInBox);
            fvec4 t = posInBox[0]*recipBoxVec0 + posInBox[1]*recipBoxVec1 + posInBox[2]*recipBoxVec2;
            t = (t-floor(t))*gridSize;
            ivec4 ti = t;
            fvec4 dr = t-ti;
            ivec4 gridIndex = ti-(gridSizeInt&ti==gridSizeInt);

            // Compute the B-spline coefficients.

            fvec4 data[PME_ORDER];
            data[PME_ORDER-1] = 0.0f;
            data[1] = dr;
            data[0] = one-dr;
            for (int j = 3; j < PME_ORDER; j++) {
                fvec4 div(1.0f/(j-1));
                data[j-1] = div*dr*data[j-2];
                for (int k = 1; k < j-1; k++)
                    data[j-k-1] = div*((dr+k)*data[j-k-2]+(fvec4(j-k)-dr)*data[j-k-1]);
                data[0] = div*(one-dr)*data[0];
            }
            data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
            for (int j = 1; j < (PME_ORDER-1); j++)
                data[PME_ORDER-j-1] = scale*((dr+j)*data[PME_ORDER-j-2]+(fvec4(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
            data[0] = scale*(one-dr)*data[0];

            // Compute the charge derivative for this atom.

            int gridIndexX = gridIndex[0];
            int gridIndexY = gridIndex[1];
            int gridIndexZ = gridIndex[2];
            if (gridIndexX < 0)
                return; // This happens when a simulation blows up and coordinates become NaN.
            int zindex[PME_ORDER];
            for (int j = 0; j < PME_ORDER; j++) {
                zindex[j] = gridIndexZ+j;
                zindex[j] -= (zindex[j] >= gridz ? gridz : 0);
            }
            float f = 0.0f;
            for (int ix = 0; ix < PME_ORDER; ix++) {
                int xbase = gridIndexX+ix;
                xbase -= (xbase >= gridx ? gridx : 0);
                xbase = xbase*gridy*gridz;
                float dx = data[ix][0];

                for (int iy = 0; iy < PME_ORDER; iy++) {
                    int ybase = gridIndexY+iy;
                    ybase -= (ybase >= gridy ? gridy : 0);
                    ybase = xbase + ybase*gridz;
                    float dy = data[iy][1];

                    for (int iz = 0; iz < PME_ORDER; iz++) {
                        float dz = data[iz][2];
                        float gridValue = grid[ybase+zindex[iz]];

                        f += dx*dy*dz*gridValue;
                    }
                }
            }

            chargeDerivatives[ii] = epsilonFactor * f;
        }
    }
}

static void* threadBody(void* args) {
    CpuCalcPmeReciprocalForceKernel& owner = *reinterpret_cast<CpuCalcPmeReciprocalForceKernel*>(args);
    owner.runMainThread();
    return 0;
}

void CpuCalcPmeReciprocalForceKernel::initialize(int xsize, int ysize, int zsize, int numParticles, const vector<int>& indices, double alpha, bool deterministic) {
    if (!hasInitializedThreads) {
        numThreads = getNumProcessors();
        char* threadsEnv = getenv("OPENMM_CPU_THREADS");
        if (threadsEnv != NULL)
            stringstream(threadsEnv) >> numThreads;
        hasInitializedThreads = true;
    }
    threadEnergy.resize(numThreads);
    gridx = findFFTDimension(xsize);
    gridy = findFFTDimension(ysize);
    gridz = findFFTDimension(zsize);
    gridShape.push_back(gridx);
    gridShape.push_back(gridy);
    gridShape.push_back(gridz);
    fftAxes.push_back(0);
    fftAxes.push_back(1);
    fftAxes.push_back(2);
    realGridStride.push_back(gridy*gridz*sizeof(float));
    realGridStride.push_back(gridz*sizeof(float));
    realGridStride.push_back(sizeof(float));
    complexGridStride.push_back(gridy*(gridz/2+1)*sizeof(complex<float>));
    complexGridStride.push_back((gridz/2+1)*sizeof(complex<float>));
    complexGridStride.push_back(sizeof(complex<float>));
    this->numParticles = numParticles;
    this->alpha = alpha;
    this->deterministic = deterministic;
    force.resize(4*numParticles);
    chargeIndices = indices;
    numIndices = chargeIndices.size();
    chargeDerivatives.resize(numIndices);
    recipEterm.resize(gridx*gridy*gridz);
    
    // Initialize threads.
    
    isFinished = false;
    mainThread = thread(threadBody, this);
    
    // Wait until the main thread is up and running.
    
    {
        unique_lock<mutex> ul(lock);
        while (!isFinished)
            endCondition.wait(ul);
    }
    
    // Initialize the FFT grids.
    //
    // X-slab partitioning: one shared real grid (+3 floats of slack for fvec4
    // unaligned reads at the boundary) and one per-thread particle list used
    // to dispatch each particle's stencil writes to the owning X-slice.

    sharedGrid.assign(gridx*gridy*gridz+3, 0.0f);
    threadParticles.resize(numThreads);
    complexGrid.resize(gridx*gridy*(gridz/2+1));

    // Initialize the b-spline moduli.

    int maxSize = std::max(std::max(gridx, gridy), gridz);
    vector<double> data(PME_ORDER);
    vector<double> ddata(PME_ORDER);
    vector<double> bsplinesData(maxSize);
    data[PME_ORDER-1] = 0.0;
    data[1] = 0.0;
    data[0] = 1.0;
    for (int i = 3; i < PME_ORDER; i++) {
        double div = 1.0/(i-1.0);
        data[i-1] = 0.0;
        for (int j = 1; j < (i-1); j++)
            data[i-j-1] = div*(j*data[i-j-2]+(i-j)*data[i-j-1]);
        data[0] = div*data[0];
    }

    // Differentiate.

    ddata[0] = -data[0];
    for (int i = 1; i < PME_ORDER; i++)
        ddata[i] = data[i-1]-data[i];
    double div = 1.0/(PME_ORDER-1);
    data[PME_ORDER-1] = 0.0;
    for (int i = 1; i < (PME_ORDER-1); i++)
        data[PME_ORDER-i-1] = div*(i*data[PME_ORDER-i-2]+(PME_ORDER-i)*data[PME_ORDER-i-1]);
    data[0] = div*data[0];
    for (int i = 0; i < maxSize; i++)
        bsplinesData[i] = 0.0;
    for (int i = 1; i <= PME_ORDER; i++)
        bsplinesData[i] = data[i-1];

    // Evaluate the actual bspline moduli for X/Y/Z.

    bsplineModuli[0].resize(gridx);
    bsplineModuli[1].resize(gridy);
    bsplineModuli[2].resize(gridz);
    for (int dim = 0; dim < 3; dim++) {
        int ndata = bsplineModuli[dim].size();
        vector<float>& moduli = bsplineModuli[dim];
        for (int i = 0; i < ndata; i++) {
            double sc = 0.0;
            double ss = 0.0;
            for (int j = 0; j < ndata; j++) {
                double arg = (2.0*M_PI*i*j)/ndata;
                sc += bsplinesData[j]*cos(arg);
                ss += bsplinesData[j]*sin(arg);
            }
            moduli[i] = (float) (sc*sc+ss*ss);
        }
        for (int i = 0; i < ndata; i++)
            if (moduli[i] < 1.0e-7f)
                moduli[i] = (moduli[(i-1+ndata)%ndata]+moduli[(i+1)%ndata])*0.5f;
    }
}

CpuCalcPmeReciprocalForceKernel::~CpuCalcPmeReciprocalForceKernel() {
    isDeleted = true;
    lock.lock();
    startCondition.notify_all();
    lock.unlock();
    mainThread.join();
}

void CpuCalcPmeReciprocalForceKernel::runMainThread() {
    // This is the main thread that coordinates all the other ones.

    unique_lock<mutex> ul(lock);
    isFinished = true;
    endCondition.notify_one();
    ThreadPool threads(numThreads);
    while (true) {
        // Wait for the signal to start.

        startCondition.wait(ul);
        if (isDeleted)
            break;
        posq = io->getPosq();
        atomicCounter = 0;
        // X-slab partitioning: bin particles into per-thread lists by which
        // X-range owner(s) their B-spline stencil touches. Doing this on the
        // main thread keeps the worker barrier count unchanged.
        buildThreadParticleLists(posq, numParticles, gridx, gridy, gridz, numThreads,
                periodicBoxVectors, recipBoxVectors, threadParticles);
        threads.execute([&] (ThreadPool& threads, int threadIndex) { runWorkerThread(threads, threadIndex); }); // Signal threads to perform charge spreading.
        threads.waitForThreads();
        // X-slab partitioning eliminates the grid-reduction phase: the shared
        // grid IS the output of spreadCharge.
        pocketfft::r2c(gridShape, realGridStride, complexGridStride, fftAxes, true, sharedGrid.data(), complexGrid.data(), 1.0f, 0);
        if (lastBoxVectors[0] != periodicBoxVectors[0] || lastBoxVectors[1] != periodicBoxVectors[1] || lastBoxVectors[2] != periodicBoxVectors[2]) {
            threads.resumeThreads(); // Signal threads to compute the reciprocal scale factors.
            threads.waitForThreads();
        }
        if (includeEnergy) {
            threads.resumeThreads(); // Signal threads to compute energy.
            threads.waitForThreads();
            for (auto e : threadEnergy)
                energy += e;
        }
        if (includeForces || includeChargeDerivatives) {
            // Explicitly zero out the zero frequency component or charge
            // derivatives will be incorrect.  The neutralizing plasma
            // interaction energy contribution is computed separately.
            complexGrid[0] = 0;

            threads.resumeThreads(); // Signal threads to perform reciprocal convolution.
            threads.waitForThreads();
            pocketfft::c2r(gridShape, complexGridStride, realGridStride, fftAxes, false, complexGrid.data(), sharedGrid.data(), 1.0f, 0);
            if (includeForces) {
                atomicCounter = 0;
                threads.resumeThreads(); // Signal threads to interpolate forces.
                threads.waitForThreads();
            }
            if (includeChargeDerivatives) {
                atomicCounter = 0;
                threads.resumeThreads(); // Signal threads to interpolate charge derivatives.
                threads.waitForThreads();
            }
        }
        threads.resumeThreads(); // Signal threads to finish.
        threads.waitForThreads();
        isFinished = true;
        lastBoxVectors[0] = periodicBoxVectors[0];
        lastBoxVectors[1] = periodicBoxVectors[1];
        lastBoxVectors[2] = periodicBoxVectors[2];
        endCondition.notify_one();
    }
}

void CpuCalcPmeReciprocalForceKernel::runWorkerThread(ThreadPool& threads, int index) {
    int gridxStart = (index*gridx)/numThreads;
    int gridxEnd = ((index+1)*gridx)/numThreads;
    int complexSize = gridx*gridy*(gridz/2+1);
    int complexStart = std::max(1, ((index*complexSize)/numThreads));
    int complexEnd = (((index+1)*complexSize)/numThreads);
    const float epsilonFactor = sqrt(ONE_4PI_EPS0);

    // X-slab partitioning: per-thread particle lists are built by the main
    // thread before execute(). Each thread zeroes its own X-slice and spreads
    // only the particles whose stencil touches that slice. No reduction step.
    spreadCharge(posq, sharedGrid, gridx, gridy, gridz, periodicBoxVectors, recipBoxVectors,
            epsilonFactor, gridxStart, gridxEnd, threadParticles[index]);
    threads.syncThreads();
    if (lastBoxVectors[0] != periodicBoxVectors[0] || lastBoxVectors[1] != periodicBoxVectors[1] || lastBoxVectors[2] != periodicBoxVectors[2]) {
        computeReciprocalEterm(gridxStart, gridxEnd, gridx, gridy, gridz, recipEterm, alpha, bsplineModuli, periodicBoxVectors, recipBoxVectors);
        threads.syncThreads();
    }
    if (includeEnergy) {
        threadEnergy[index] = reciprocalEnergy(gridxStart, gridxEnd, complexGrid, recipEterm, gridx, gridy, gridz, alpha, bsplineModuli, periodicBoxVectors, recipBoxVectors);
        threads.syncThreads();
    }
    if (includeForces || includeChargeDerivatives) {
        reciprocalConvolution(complexStart, complexEnd, complexGrid, recipEterm);
        threads.syncThreads();
        if (includeForces) {
            interpolateForces(posq, force, sharedGrid, gridx, gridy, gridz, numParticles, periodicBoxVectors, recipBoxVectors, atomicCounter, epsilonFactor, numThreads);
            threads.syncThreads();
        }
        if (includeChargeDerivatives) {
            interpolateChargeDerivatives(posq, chargeIndices, chargeDerivatives, sharedGrid, gridx, gridy, gridz, numIndices, periodicBoxVectors, recipBoxVectors, atomicCounter, epsilonFactor, numThreads);
            threads.syncThreads();
        }
    }
}

void CpuCalcPmeReciprocalForceKernel::beginComputation(IO& io, const Vec3* periodicBoxVectors, bool includeEnergy, bool includeForces, bool includeChargeDerivatives) {
    this->io = &io;
    this->periodicBoxVectors[0] = periodicBoxVectors[0];
    this->periodicBoxVectors[1] = periodicBoxVectors[1];
    this->periodicBoxVectors[2] = periodicBoxVectors[2];
    this->includeEnergy = includeEnergy;
    this->includeForces = includeForces;
    this->includeChargeDerivatives = includeChargeDerivatives;
    energy = 0.0;
    ReferenceForce::invertBoxVectors(periodicBoxVectors, recipBoxVectors);

    // Do the calculation.

    unique_lock<mutex> ul(lock);
    isFinished = false;
    startCondition.notify_one();
}

double CpuCalcPmeReciprocalForceKernel::finishComputation(IO& io) {
    {
        unique_lock<mutex> ul(lock);
        while (!isFinished) {
            endCondition.wait(ul);
        }
    }
    if (includeForces) {
        io.setForce(&force[0]);
    }
    if (includeChargeDerivatives) {
        io.setChargeDerivatives(&chargeDerivatives[0]);
    }
    return energy;
}

bool CpuCalcPmeReciprocalForceKernel::isProcessorSupported() {
    return isVec4Supported();
}

void CpuCalcPmeReciprocalForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = gridx;
    ny = gridy;
    nz = gridz;
}

int CpuCalcPmeReciprocalForceKernel::findFFTDimension(int minimum) {
    if (minimum < 1)
        return 1;
    while (true) {
        // Attempt to factor the current value.

        int unfactored = minimum;
        for (int factor = 2; factor < 9; factor++) {
            while (unfactored > 1 && unfactored%factor == 0)
                unfactored /= factor;
        }
        if (unfactored == 1 || unfactored == 11)
            return minimum;
        minimum++;
    }
}

/*
 * Everything below here is just a clone of the above, but to handle the dispersion term
 * instead of electrostatics.
 */

bool CpuCalcPmeReciprocalForceKernel::hasInitializedThreads = false;
int CpuCalcPmeReciprocalForceKernel::numThreads = 0;


class CpuCalcDispersionPmeReciprocalForceKernel::ComputeTask : public ThreadPool::Task {
public:
    ComputeTask(CpuCalcDispersionPmeReciprocalForceKernel& owner) : owner(owner) {
    }
    void execute(ThreadPool& threads, int threadIndex) {
        owner.runWorkerThread(threads, threadIndex);
    }
    CpuCalcDispersionPmeReciprocalForceKernel& owner;
};

static void* dispersionThreadBody(void* args) {
    CpuCalcDispersionPmeReciprocalForceKernel& owner = *reinterpret_cast<CpuCalcDispersionPmeReciprocalForceKernel*>(args);
    owner.runMainThread();
    return 0;
}

void CpuCalcDispersionPmeReciprocalForceKernel::initialize(int xsize, int ysize, int zsize, int numParticles, double alpha, bool deterministic) {
    if (!hasInitializedThreads) {
        numThreads = getNumProcessors();
        char* threadsEnv = getenv("OPENMM_CPU_THREADS");
        if (threadsEnv != NULL)
            stringstream(threadsEnv) >> numThreads;
        hasInitializedThreads = true;
    }
    threadEnergy.resize(numThreads);
    gridx = findFFTDimension(xsize);
    gridy = findFFTDimension(ysize);
    gridz = findFFTDimension(zsize);
    gridShape.push_back(gridx);
    gridShape.push_back(gridy);
    gridShape.push_back(gridz);
    fftAxes.push_back(0);
    fftAxes.push_back(1);
    fftAxes.push_back(2);
    realGridStride.push_back(gridy*gridz*sizeof(float));
    realGridStride.push_back(gridz*sizeof(float));
    realGridStride.push_back(sizeof(float));
    complexGridStride.push_back(gridy*(gridz/2+1)*sizeof(complex<float>));
    complexGridStride.push_back((gridz/2+1)*sizeof(complex<float>));
    complexGridStride.push_back(sizeof(complex<float>));
    this->numParticles = numParticles;
    this->alpha = alpha;
    this->deterministic = deterministic;
    force.resize(4*numParticles);
    recipEterm.resize(gridx*gridy*gridz);
    
    // Initialize threads.
    
    isFinished = false;
    mainThread = thread(dispersionThreadBody, this);
    
    // Wait until the main thread is up and running.
    
    {
        unique_lock<mutex> ul(lock);
        while (!isFinished)
            endCondition.wait(ul);
    }

    // Initialize the FFT grids (X-slab partitioned: a single shared grid).

    sharedGrid.assign(gridx*gridy*gridz+3, 0.0f);
    threadParticles.resize(numThreads);
    complexGrid.resize(gridx*gridy*(gridz/2+1));

    // Initialize the b-spline moduli.

    int maxSize = std::max(std::max(gridx, gridy), gridz);
    vector<double> data(PME_ORDER);
    vector<double> ddata(PME_ORDER);
    vector<double> bsplinesData(maxSize);
    data[PME_ORDER-1] = 0.0;
    data[1] = 0.0;
    data[0] = 1.0;
    for (int i = 3; i < PME_ORDER; i++) {
        double div = 1.0/(i-1.0);
        data[i-1] = 0.0;
        for (int j = 1; j < (i-1); j++)
            data[i-j-1] = div*(j*data[i-j-2]+(i-j)*data[i-j-1]);
        data[0] = div*data[0];
    }

    // Differentiate.

    ddata[0] = -data[0];
    for (int i = 1; i < PME_ORDER; i++)
        ddata[i] = data[i-1]-data[i];
    double div = 1.0/(PME_ORDER-1);
    data[PME_ORDER-1] = 0.0;
    for (int i = 1; i < (PME_ORDER-1); i++)
        data[PME_ORDER-i-1] = div*(i*data[PME_ORDER-i-2]+(PME_ORDER-i)*data[PME_ORDER-i-1]);
    data[0] = div*data[0];
    for (int i = 0; i < maxSize; i++)
        bsplinesData[i] = 0.0;
    for (int i = 1; i <= PME_ORDER; i++)
        bsplinesData[i] = data[i-1];

    // Evaluate the actual bspline moduli for X/Y/Z.

    bsplineModuli[0].resize(gridx);
    bsplineModuli[1].resize(gridy);
    bsplineModuli[2].resize(gridz);
    for (int dim = 0; dim < 3; dim++) {
        int ndata = bsplineModuli[dim].size();
        vector<float>& moduli = bsplineModuli[dim];
        for (int i = 0; i < ndata; i++) {
            double sc = 0.0;
            double ss = 0.0;
            for (int j = 0; j < ndata; j++) {
                double arg = (2.0*M_PI*i*j)/ndata;
                sc += bsplinesData[j]*cos(arg);
                ss += bsplinesData[j]*sin(arg);
            }
            moduli[i] = (float) (sc*sc+ss*ss);
        }
        for (int i = 0; i < ndata; i++)
            if (moduli[i] < 1.0e-7f)
                moduli[i] = (moduli[i-1]+moduli[i+1])*0.5f;
    }
}

CpuCalcDispersionPmeReciprocalForceKernel::~CpuCalcDispersionPmeReciprocalForceKernel() {
    isDeleted = true;
    lock.lock();
    startCondition.notify_all();
    lock.unlock();
    mainThread.join();
}

void CpuCalcDispersionPmeReciprocalForceKernel::runMainThread() {
    // This is the main thread that coordinates all the other ones.

    unique_lock<mutex> ul(lock);
    isFinished = true;
    endCondition.notify_one();
    ThreadPool threads(numThreads);
    while (true) {
        // Wait for the signal to start.

        startCondition.wait(ul);
        if (isDeleted)
            break;
        posq = io->getPosq();
        ComputeTask task(*this);
        atomicCounter = 0;
        // X-slab partitioning: build per-thread particle bins on the main
        // thread so the per-worker barrier count is unchanged.
        buildThreadParticleLists(posq, numParticles, gridx, gridy, gridz, numThreads,
                periodicBoxVectors, recipBoxVectors, threadParticles);
        threads.execute(task); // Signal threads to perform charge spreading.
        threads.waitForThreads();
        // No reduction step: the shared grid is the spread output.
        pocketfft::r2c(gridShape, realGridStride, complexGridStride, fftAxes, true, sharedGrid.data(), complexGrid.data(), 1.0f, 0);
        if (lastBoxVectors[0] != periodicBoxVectors[0] || lastBoxVectors[1] != periodicBoxVectors[1] || lastBoxVectors[2] != periodicBoxVectors[2]) {
            threads.resumeThreads(); // Signal threads to compute the reciprocal scale factors.
            threads.waitForThreads();
        }
        if (includeEnergy) {
            threads.resumeThreads(); // Signal threads to compute energy.
            threads.waitForThreads();
            for (auto e : threadEnergy)
                energy += e;
        }
        threads.resumeThreads(); // Signal threads to perform reciprocal convolution.
        threads.waitForThreads();
        pocketfft::c2r(gridShape, complexGridStride, realGridStride, fftAxes, false, complexGrid.data(), sharedGrid.data(), 1.0f, 0);
        atomicCounter = 0;
        threads.resumeThreads(); // Signal threads to interpolate forces.
        threads.waitForThreads();
        isFinished = true;
        lastBoxVectors[0] = periodicBoxVectors[0];
        lastBoxVectors[1] = periodicBoxVectors[1];
        lastBoxVectors[2] = periodicBoxVectors[2];
        endCondition.notify_one();
    }
}

void CpuCalcDispersionPmeReciprocalForceKernel::runWorkerThread(ThreadPool& threads, int index) {
    int gridxStart = (index*gridx)/numThreads;
    int gridxEnd = ((index+1)*gridx)/numThreads;
    int complexSize = gridx*gridy*(gridz/2+1);
    int complexStart = std::max(1, ((index*complexSize)/numThreads));
    int complexEnd = (((index+1)*complexSize)/numThreads);
    const float epsilonFactor = 1.0f;
    // X-slab partitioning: per-thread particle lists are built on the main
    // thread before execute(). Each worker zeroes its own X-slice and spreads
    // only the particles whose stencil touches that slice. No reduction.
    spreadCharge(posq, sharedGrid, gridx, gridy, gridz, periodicBoxVectors, recipBoxVectors,
            epsilonFactor, gridxStart, gridxEnd, threadParticles[index]);
    threads.syncThreads();
    if (lastBoxVectors[0] != periodicBoxVectors[0] || lastBoxVectors[1] != periodicBoxVectors[1] || lastBoxVectors[2] != periodicBoxVectors[2]) {
        computeReciprocalDispersionEterm(gridxStart, gridxEnd, gridx, gridy, gridz, recipEterm, alpha, bsplineModuli, periodicBoxVectors, recipBoxVectors);
        threads.syncThreads();
    }
    if (includeEnergy) {
        threadEnergy[index] = reciprocalDispersionEnergy(gridxStart, gridxEnd, complexGrid, recipEterm, gridx, gridy, gridz, alpha, bsplineModuli, periodicBoxVectors, recipBoxVectors);
        threads.syncThreads();
    }
    // For dispersion, we include the {0,0,0} term, so the start point needs to be redefined
    complexStart = (index*complexSize)/numThreads;
    reciprocalConvolution(complexStart, complexEnd, complexGrid, recipEterm);
    threads.syncThreads();
    interpolateForces(posq, force, sharedGrid, gridx, gridy, gridz, numParticles, periodicBoxVectors, recipBoxVectors, atomicCounter, epsilonFactor, numThreads);
}

void CpuCalcDispersionPmeReciprocalForceKernel::beginComputation(CalcPmeReciprocalForceKernel::IO& io, const Vec3* periodicBoxVectors, bool includeEnergy) {
    this->io = &io;
    this->periodicBoxVectors[0] = periodicBoxVectors[0];
    this->periodicBoxVectors[1] = periodicBoxVectors[1];
    this->periodicBoxVectors[2] = periodicBoxVectors[2];
    this->includeEnergy = includeEnergy;
    energy = 0.0;
    ReferenceForce::invertBoxVectors(periodicBoxVectors, recipBoxVectors);

    // Do the calculation.

    unique_lock<mutex> ul(lock);
    isFinished = false;
    startCondition.notify_one();
}

double CpuCalcDispersionPmeReciprocalForceKernel::finishComputation(CalcPmeReciprocalForceKernel::IO& io) {
    {
        unique_lock<mutex> ul(lock);
        while (!isFinished) {
            endCondition.wait(ul);
        }
    }
    io.setForce(&force[0]);
    return energy;
}

bool CpuCalcDispersionPmeReciprocalForceKernel::isProcessorSupported() {
    return isVec4Supported();
}

void CpuCalcDispersionPmeReciprocalForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = gridx;
    ny = gridy;
    nz = gridz;
}

int CpuCalcDispersionPmeReciprocalForceKernel::findFFTDimension(int minimum) {
    if (minimum < 1)
        return 1;
    while (true) {
        // Attempt to factor the current value.

        int unfactored = minimum;
        for (int factor = 2; factor < 9; factor++) {
            while (unfactored > 1 && unfactored%factor == 0)
                unfactored /= factor;
        }
        if (unfactored == 1 || unfactored == 11)
            return minimum;
        minimum++;
    }
}
