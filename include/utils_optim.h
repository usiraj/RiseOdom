/*
 * utils_optim.h
 *
 *  Created on: March 11, 2015
 *      Author: usama
 */
#ifndef UTILS_OPTIM_H
#define UTILS_OPTIM_H

#include <opencv2/opencv.hpp>
#include "tbb/tbb.h"
#include "utils.h"

#define FASTHAMMING_FREAK_PARSCORE 32	// 44/128

#if defined(_MSC_VER)
     /* Microsoft C/C++-compatible compiler */
     #include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
     /* GCC-compatible compiler, targeting x86/x86-64 */
     #include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
     /* GCC-compatible compiler, targeting ARM with NEON */
     #include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
     /* GCC-compatible compiler, targeting ARM with WMMX */
     #include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
     /* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
     #include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
     /* GCC-compatible compiler, targeting PowerPC with SPE */
     #include <spe.h>
#endif
void fast_hammingnorm_matchserial(const cv::Mat &des1, const cv::Mat &des2, ListIdxCorrespondance &_list, const float ratio=0.45);
void fast_hammingnorm_match(const cv::Mat &des1, const cv::Mat&des2, ListIdxCorrespondance &_list, const float ratio=0.45,
	const int par_score=FASTHAMMING_FREAK_PARSCORE);
void fast_hammingnorm_match_neighborhood(const cv::Mat &des1, const cv::Mat&des2, const cv::Mat &neighborhood,
										 ListIdxCorrespondance &_list, const float ratio=0.45,
										 const int par_score=FASTHAMMING_FREAK_PARSCORE);
int fast_hammingnorm(const unsigned char* a, const unsigned char* b, const int n);
// some other functions
int _normoptimized(const unsigned char* a, const unsigned char* b, const int n);
int _normoptimized16(const unsigned char* a, const unsigned char* b);
int _normoptimized32(const unsigned char* a, const unsigned char* b);
int _normoptimized48(const unsigned char* a, const unsigned char* b);
int _normoptimized64(const unsigned char*a, const unsigned char*b);
#endif