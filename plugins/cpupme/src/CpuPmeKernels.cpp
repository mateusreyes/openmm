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

// Cache-line-granularity touched bitmap. One bit per CELLS_PER_LINE cells
// (= one bit per 64-byte cache line on x86). On first write to a line, the
// whole line is eagerly zeroed in SIMD and the bit is set; subsequent
// writes to the line just load+add+store. This makes the per-scatter
// bookkeeping a single bit check / branch instead of the bit-level
// shift+mask+update used by the per-cell bitset, and shrinks the bitmap
// 16x so it fits comfortably in L1.
static const int CELLS_PER_LINE = 16; // 64 B / sizeof(float)

// Size of the per-thread touched bitmap, in bytes.
static inline size_t touchedBitmapSize(int gridx, int gridy, int gridz) {
    size_t numLines = ((size_t)gridx*gridy*gridz + CELLS_PER_LINE - 1) / CELLS_PER_LINE;
    return (numLines + 7) / 8 + 1;
}

// Grid storage size, rounded up so that line-aligned zeroing never runs
// past the end of the buffer. +3 of trailing pad for the existing
// unaligned fvec4 read pattern.
static inline size_t paddedGridSize(int gridx, int gridy, int gridz) {
    size_t total = (size_t)gridx*gridy*gridz;
    return ((total + CELLS_PER_LINE - 1) / CELLS_PER_LINE) * CELLS_PER_LINE + 3;
}

// Zero out one 16-cell cache line (4 fvec4 stores).
static inline void zeroLine(float* grid, int line_base) {
    fvec4 z(0.0f);
    z.store(&grid[line_base + 0]);
    z.store(&grid[line_base + 4]);
    z.store(&grid[line_base + 8]);
    z.store(&grid[line_base + 12]);
}

// Scatter-add a 4-lane SIMD vector into grid at consecutive index idx.
// The (very common) fast path stays in a single cache line and pays
// only one branch on the touched bit.
static inline void scatter_add4(float* grid, uint8_t* touched, int idx, fvec4 add);
static inline void scatter_add1(float* grid, uint8_t* touched, int idx, float add) {
    int line = idx / CELLS_PER_LINE;
    int line_byte = line >> 3;
    uint8_t bit = (uint8_t)(1 << (line & 7));
    if (!(touched[line_byte] & bit)) {
        zeroLine(grid, line * CELLS_PER_LINE);
        touched[line_byte] = (uint8_t)(touched[line_byte] | bit);
    }
    grid[idx] += add;
}
static inline void scatter_add4(float* grid, uint8_t* touched, int idx, fvec4 add) {
    int line = idx / CELLS_PER_LINE;
    int offset = idx - line * CELLS_PER_LINE;
    if (offset + 4 <= CELLS_PER_LINE) {
        // Fast path: all 4 lanes inside one cache line.
        int line_byte = line >> 3;
        uint8_t bit = (uint8_t)(1 << (line & 7));
        if (!(touched[line_byte] & bit)) {
            zeroLine(grid, line * CELLS_PER_LINE);
            touched[line_byte] = (uint8_t)(touched[line_byte] | bit);
        }
        (fvec4(&grid[idx]) + add).store(&grid[idx]);
    } else {
        // Rare: 4 lanes span 2 cache lines. Fall back to scalar.
        float v[4];
        add.store(v);
        scatter_add1(grid, touched, idx + 0, v[0]);
        scatter_add1(grid, touched, idx + 1, v[1]);
        scatter_add1(grid, touched, idx + 2, v[2]);
        scatter_add1(grid, touched, idx + 3, v[3]);
    }
}

bool CpuCalcDispersionPmeReciprocalForceKernel::hasInitializedThreads = false;
int CpuCalcDispersionPmeReciprocalForceKernel::numThreads = 0;

static void spreadCharge(float* posq, vector<float>& grid, vector<uint8_t>& touched, int gridx, int gridy, int gridz, int numParticles, Vec3* periodicBoxVectors, Vec3* recipBoxVectors,
        atomic<int>& atomicCounter, const float epsilonFactor, int threadIndex, int numThreads, bool deterministic) {
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
    // Clear only the touched-bitmap (32x smaller than the grid). The
    // grid itself is left holding stale values from the prior step; the
    // bitmap encodes "logical zero" so scatter reads mask stale data.
    float* gridPtr = grid.data();
    uint8_t* touchedPtr = touched.data();
    memset(touchedPtr, 0, touched.size());

    const int groupSize = max(1, numParticles / (10 * numThreads));
    int start = groupSize * threadIndex;
    while (true) {
        if (!deterministic)
            start = atomicCounter.fetch_add(groupSize);

        if (start >= numParticles)
            break;

        int end = min(start + groupSize, numParticles);
        for (int i = start; i < end; ++i) {
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
                    int xbase = gridIndexX+ix;
                    xbase -= (xbase >= gridx ? gridx : 0);
                    xbase = xbase*gridy*gridz;
                    float xdata = charge*data[ix][0];
                    for (int iy = 0; iy < PME_ORDER; iy++) {
                        int ybase = gridIndexY+iy;
                        ybase -= (ybase >= gridy ? gridy : 0);
                        ybase = xbase + ybase*gridz;
                        float multiplier = xdata*data[iy][1];
                        fvec4 add0to3 = zdata0to3*multiplier;
                        scatter_add4(gridPtr, touchedPtr, ybase+gridIndexZ, add0to3);
                        scatter_add1(gridPtr, touchedPtr, ybase+zindex[4], multiplier*zdata4);
                    }
                }
            }
            else {
                for (int ix = 0; ix < PME_ORDER; ix++) {
                    int xbase = gridIndexX+ix;
                    xbase -= (xbase >= gridx ? gridx : 0);
                    xbase = xbase*gridy*gridz;
                    float xdata = charge*data[ix][0];
                    for (int iy = 0; iy < PME_ORDER; iy++) {
                        int ybase = gridIndexY+iy;
                        ybase -= (ybase >= gridy ? gridy : 0);
                        ybase = xbase + ybase*gridz;
                        float multiplier = xdata*data[iy][1];
                        fvec4 add0to3 = zdata0to3*multiplier;
                        add0to3.store(temp);
                        scatter_add1(gridPtr, touchedPtr, ybase+zindex[0], temp[0]);
                        scatter_add1(gridPtr, touchedPtr, ybase+zindex[1], temp[1]);
                        scatter_add1(gridPtr, touchedPtr, ybase+zindex[2], temp[2]);
                        scatter_add1(gridPtr, touchedPtr, ybase+zindex[3], temp[3]);
                        scatter_add1(gridPtr, touchedPtr, ybase+zindex[4], multiplier*zdata4);
                    }
                }
            }
        }

        if (deterministic)
            start += groupSize * numThreads;
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

    realGrids.resize(numThreads, vector<float>(paddedGridSize(gridx, gridy, gridz)));
    touchedGrids.resize(numThreads, vector<uint8_t>(touchedBitmapSize(gridx, gridy, gridz)));
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
        threads.execute([&] (ThreadPool& threads, int threadIndex) { runWorkerThread(threads, threadIndex); }); // Signal threads to perform charge spreading.
        threads.waitForThreads();
        threads.resumeThreads(); // Signal threads to sum the charge grids.
        threads.waitForThreads();
        pocketfft::r2c(gridShape, realGridStride, complexGridStride, fftAxes, true, realGrids[0].data(), complexGrid.data(), 1.0f, 0);
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
            pocketfft::c2r(gridShape, complexGridStride, realGridStride, fftAxes, false, complexGrid.data(), realGrids[0].data(), 1.0f, 0);
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
    int numLines = (int)(paddedGridSize(gridx, gridy, gridz) - 3) / CELLS_PER_LINE;
    int lineStart = (index*numLines)/numThreads;
    int lineEnd = ((index+1)*numLines)/numThreads;
    int complexSize = gridx*gridy*(gridz/2+1);
    int complexStart = std::max(1, ((index*complexSize)/numThreads));
    int complexEnd = (((index+1)*complexSize)/numThreads);
    const float epsilonFactor = sqrt(ONE_4PI_EPS0);
    spreadCharge(posq, realGrids[index], touchedGrids[index], gridx, gridy, gridz, numParticles, periodicBoxVectors, recipBoxVectors, atomicCounter, epsilonFactor, index, numThreads, deterministic);
    threads.syncThreads();
    int numGrids = realGrids.size();
    // Reduce per-thread grids into realGrids[0] one cache line at a time.
    // Untouched lines contribute zero (sum stays at 0 before the store),
    // so stale data from prior steps never enters the FFT input.
    for (int line = lineStart; line < lineEnd; line++) {
        int base = line * CELLS_PER_LINE;
        int byte = line >> 3;
        uint8_t bit = (uint8_t)(1 << (line & 7));
        fvec4 s0(0.0f), s1(0.0f), s2(0.0f), s3(0.0f);
        for (int j = 0; j < numGrids; j++) {
            if (touchedGrids[j][byte] & bit) {
                s0 = s0 + fvec4(&realGrids[j][base + 0]);
                s1 = s1 + fvec4(&realGrids[j][base + 4]);
                s2 = s2 + fvec4(&realGrids[j][base + 8]);
                s3 = s3 + fvec4(&realGrids[j][base + 12]);
            }
        }
        s0.store(&realGrids[0][base + 0]);
        s1.store(&realGrids[0][base + 4]);
        s2.store(&realGrids[0][base + 8]);
        s3.store(&realGrids[0][base + 12]);
    }
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
            interpolateForces(posq, force, realGrids[0], gridx, gridy, gridz, numParticles, periodicBoxVectors, recipBoxVectors, atomicCounter, epsilonFactor, numThreads);
            threads.syncThreads();
        }
        if (includeChargeDerivatives) {
            interpolateChargeDerivatives(posq, chargeIndices, chargeDerivatives, realGrids[0], gridx, gridy, gridz, numIndices, periodicBoxVectors, recipBoxVectors, atomicCounter, epsilonFactor, numThreads);
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

    // Initialize the FFT grids.

    realGrids.resize(numThreads, vector<float>(paddedGridSize(gridx, gridy, gridz)));
    touchedGrids.resize(numThreads, vector<uint8_t>(touchedBitmapSize(gridx, gridy, gridz)));
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
        threads.execute(task); // Signal threads to perform charge spreading.
        threads.waitForThreads();
        threads.resumeThreads(); // Signal threads to sum the charge grids.
        threads.waitForThreads();
        pocketfft::r2c(gridShape, realGridStride, complexGridStride, fftAxes, true, realGrids[0].data(), complexGrid.data(), 1.0f, 0);
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
        pocketfft::c2r(gridShape, complexGridStride, realGridStride, fftAxes, false, complexGrid.data(), realGrids[0].data(), 1.0f, 0);
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
    int numLines = (int)(paddedGridSize(gridx, gridy, gridz) - 3) / CELLS_PER_LINE;
    int lineStart = (index*numLines)/numThreads;
    int lineEnd = ((index+1)*numLines)/numThreads;
    int complexSize = gridx*gridy*(gridz/2+1);
    int complexStart = std::max(1, ((index*complexSize)/numThreads));
    int complexEnd = (((index+1)*complexSize)/numThreads);
    const float epsilonFactor = 1.0f;
    spreadCharge(posq, realGrids[index], touchedGrids[index], gridx, gridy, gridz, numParticles, periodicBoxVectors, recipBoxVectors, atomicCounter, epsilonFactor, index, numThreads, deterministic);
    threads.syncThreads();
    int numGrids = realGrids.size();
    // Reduce per-thread grids into realGrids[0] one cache line at a time;
    // see CpuCalcPmeReciprocalForceKernel::runWorkerThread for details.
    for (int line = lineStart; line < lineEnd; line++) {
        int base = line * CELLS_PER_LINE;
        int byte = line >> 3;
        uint8_t bit = (uint8_t)(1 << (line & 7));
        fvec4 s0(0.0f), s1(0.0f), s2(0.0f), s3(0.0f);
        for (int j = 0; j < numGrids; j++) {
            if (touchedGrids[j][byte] & bit) {
                s0 = s0 + fvec4(&realGrids[j][base + 0]);
                s1 = s1 + fvec4(&realGrids[j][base + 4]);
                s2 = s2 + fvec4(&realGrids[j][base + 8]);
                s3 = s3 + fvec4(&realGrids[j][base + 12]);
            }
        }
        s0.store(&realGrids[0][base + 0]);
        s1.store(&realGrids[0][base + 4]);
        s2.store(&realGrids[0][base + 8]);
        s3.store(&realGrids[0][base + 12]);
    }
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
    interpolateForces(posq, force, realGrids[0], gridx, gridy, gridz, numParticles, periodicBoxVectors, recipBoxVectors, atomicCounter, epsilonFactor, numThreads);
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
