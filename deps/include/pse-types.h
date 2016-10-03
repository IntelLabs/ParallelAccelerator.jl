/*
Copyright (c) 2015, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PSE_TYPES_H_
#define PSE_TYPES_H_

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <iostream>
#include <sstream>
#include <algorithm>

static unsigned cur_threads_used = 1;

#ifdef _OPENMP
#include <omp.h>
#endif

class J2cParRegionThreadCount {
protected:
    unsigned num_threads_used;
    unsigned m_line;
    const char *   m_file;
    unsigned m_host_min_par;
    unsigned m_phi_min_par;
public:
    J2cParRegionThreadCount(unsigned iteration_count, unsigned line, const char *file, unsigned host_min = 0, unsigned phi_min = 0) :
        m_line(line),
        m_file(file),
        m_host_min_par(host_min),
        m_phi_min_par(phi_min) {
#ifdef _OPENMP
        unsigned max = omp_get_max_threads(); // the max number of threads for this device
#else
        unsigned max = 1;
#endif

        // Don't oversubscribe too much
        if (cur_threads_used >= max) {
#ifdef DEBUGJ2C
            printf("%s %d %d %d enter max =%d thread %d\n", file, line, map, _Offload_get_device_number(), max, cur_threads_used);
#endif
            num_threads_used = 1;
        } else {
            // how many unused threads are there
            unsigned num_left = max - cur_threads_used;
            // assuming all threads are doing the same thing, take the num free and divide by how many cohorts there are to get our fair share of the rest
            unsigned our_share = (num_left / cur_threads_used) + 1;
            // never allocate more thread to this loop than the size of the loop
            num_threads_used = std::min(our_share, iteration_count);
#if 0
            if (num_threads_used < 2) {
                num_threads_used = 2;
            }
#endif
#ifdef DEBUGJ2C
            printf("%s %d %d %d enter ic = %d max = %d num_left = %d our_share = %d ntu = %d\n", file, line, max, _Offload_get_device_number(), iteration_count, max, num_left, our_share, num_threads_used);
#endif
        }

        if(num_threads_used > 1 && runInPar()) {
            // update the global
            unsigned prev = __sync_fetch_and_add(&cur_threads_used, num_threads_used);
#ifdef DEBUGJ2C
            printf("%s %d %d %d enter prev = %d new = %d\n", file, line, max, _Offload_get_device_number(), prev, prev + num_threads_used);
#endif
        }
    }

    ~J2cParRegionThreadCount(void) {
        if(num_threads_used > 1 && runInPar()) {
            unsigned prev = __sync_fetch_and_sub(&cur_threads_used, num_threads_used);
#ifdef DEBUGJ2C
            printf("%s %d %d %d exit prev = %d new = %d\n", m_file, m_line, max, _Offload_get_device_number(), prev, prev - num_threads_used);
#endif
        } else {
#ifdef DEBUGJ2C
          //  printf("%s %d %d %d exit\n", m_file, m_line, omp_get_thread_num(), _Offload_get_device_number());
#endif
        }
    }

    unsigned getUsed(void) const {
        return num_threads_used;
    }

    bool runInPar(void) const {
#ifdef __MIC__
        return num_threads_used >= m_phi_min_par;
#else
        return num_threads_used >= m_host_min_par;
#endif
    }
};

unsigned computeNumThreads(uint64_t instruction_count_estimate) {
#ifdef __MIC__
    unsigned est = instruction_count_estimate / 5500000;
#else
    unsigned est = instruction_count_estimate / 21000000;
#endif
#ifdef _OPENMP
    int max = omp_get_max_threads();
#else
    int max = 1;
#endif
    unsigned ret = est > max ? max : est == 0 ? 1 : est;
//    printf("computeNumThreads: %lld => %d\n", instruction_count_estimate, ret);
    return ret;
}

#define ALLOC alloc_if(1) free_if(0)
#define FREE  alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)

#include <sys/time.h>

#ifdef TARGET_ATTRIBUTE
TARGET_ATTRIBUTE
#endif
static double timestamp ()
{
	struct timeval tv;
	gettimeofday (&tv, 0);
	return tv.tv_sec + 1e-6*tv.tv_usec;
}

#define julia_Base__assert(x) assert(x)
#define julia_Base__StepRange(x,y,z) StepRange{x,y,z}

int64_t checked_sadd(int64_t a, int64_t b) {
    if (a > 0 && b > LLONG_MAX - a) {
        /* handle overflow */
        assert(0);
    } else if (a < 0 && b < LLONG_MIN - a) {
        /* handle underflow */
        assert(0);
    }
    return a + b;
}

#define checked_ssub(a, b) checked_sadd(a,-b)

#endif /* PSE_TYPES_H_ */
