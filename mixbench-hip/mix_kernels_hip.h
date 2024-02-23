/**
 * mix_kernels_hip.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#pragma once

extern int g_loop;

extern int g_cal_sp ;
extern int g_cal_sp2;
extern int g_cal_dp ;
extern int g_cal_hp ;
extern int g_cal_int;

extern double g_computations_sp ;
extern double g_computations_sp2;
extern double g_computations_dp ;
extern double g_computations_hp ;
extern double g_computations_int;

extern float g_kernel_time_mad_sp_t ;
extern float g_kernel_time_mad_sp2_t;
extern float g_kernel_time_mad_dp_t ;
extern float g_kernel_time_mad_hp_t ;
extern float g_kernel_time_mad_int_t;

extern "C" void mixbenchGPU(double*, long size);

