/**********
Copyright (c) 2020, Xilinx, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce latency and 
    device resource utilization of the resulting RTL code
    This is vector addition example to demonstrate how HLS optimizations are used in kernel. 
*******************************************************************************/


#include <stdio.h>
#include <iostream>
//#include <cstdlib.h>

// Maximum Array Size
#define FIXED_N 16

// TRIPCOUNT identifier
const unsigned int c_size = FIXED_N;

int num_of_kernel_calls = 1;

int flag = 1;

using namespace std;

extern "C" {
void multadd(const float* a, // Read-Only Matrix A
           const float* b, // Read-Only Matrix B
           float* c,       // Output Result
           int a_row,    // Matrix A Row Size
           int a_col,    // Matrix A Col Size
           int b_col,     // Matrix B Col Size // Col size of A is equal to row size of B
           int block_count    /////////////Project
           ) {

    std::cout<<"\nKernel ---------- Number of Kernel Calls: "<<num_of_kernel_calls++<<"\n";

    #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem

    #pragma HLS INTERFACE s_axilite port=a bundle=control
    #pragma HLS INTERFACE s_axilite port=b bundle=control
    #pragma HLS INTERFACE s_axilite port=c bundle=control
    #pragma HLS INTERFACE s_axilite port=a_row bundle=control
    #pragma HLS INTERFACE s_axilite port=a_col bundle=control
    #pragma HLS INTERFACE s_axilite port=b_col bundle=control
    #pragma HLS INTERFACE s_axilite port=block_count bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int b_row = a_col;
    int c_row = a_row;
    int c_col = b_col;

    // Local memory to store input and output matrices
    float localA[FIXED_N][FIXED_N];
#pragma HLS ARRAY_PARTITION variable = localA dim = 1 complete

    float localB[FIXED_N][FIXED_N];
#pragma HLS ARRAY_PARTITION variable = localB dim = 2 complete

    float localC[FIXED_N][FIXED_N];
#pragma HLS ARRAY_PARTITION variable = localC dim = 0 complete


////////// for accumulating
    float C_temp[FIXED_N * FIXED_N];
/////////////Project////////////////////////////////////////////////////////

// Burst reads on input matrices from global memory
// Read Input A
// Auto-pipeline is going to apply pipeline to these loops
readA:
    for (int loc = 0, i = 0, j = 0; loc < a_row * a_col; loc++, j++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size* c_size max = c_size * c_size
        if (j == a_col) {
            i++;
            j = 0;
        }
        localA[i][j] = a[loc];
    }

// Read Input B
readB:
    for (int loc = 0, i = 0, j = 0; loc < b_row * b_col; loc++, j++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size* c_size max = c_size * c_size
        if (j == b_col) {
            i++;
            j = 0;
        }
        localB[i][j] = b[loc];
    }

systolic1:
    for (int k = 0; k < a_col; k++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
    systolic2:
        for (int i = 0; i < FIXED_N; i++) {
#pragma HLS UNROLL
        systolic3:
            for (int j = 0; j < FIXED_N; j++) {
#pragma HLS UNROLL
                // Get previous C_temp
                int last = (k == 0) ? 0 : localC[i][j];

                // Update current C_temp
                // Handle boundary conditions
                float a_val = (i < a_row && k < a_col) ? localA[i][k] : 0;
                float b_val = (k < b_row && j < b_col) ? localB[k][j] : 0;
                float result = last + a_val * b_val;

                // Write back results
                localC[i][j] = result;
            }
        }
    }

// Burst write from output matrices to global memory
// Burst write from matrix C

/////////////Addition in the kernel
for (int loc = 0, i = 0, j = 0; loc < c_row * c_col; loc++, j++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size* c_size max = c_size * c_size
        if (j == c_col) {
            i++;
            j = 0;
        }

        if(flag == 1) // First time the kernel is called, to avoid junk value in C_temp
        {
            C_temp[loc] = localC[i][j];
        }
        else
            C_temp[loc] += localC[i][j]; // accumulate AijBji partial products
    }
    flag = 0; //once first call is done

std::cout<<"\nKernel ---------- C_temp: "<<C_temp[0]<<"\n";

// for (int i = 0; i < FIXED_N; i++)
// {
//     for (int j = 0; j < FIXED_N; j++)
//     {
//         std::cout<<C_temp[i + j * FIXED_N];
//     }
//     std::cout<<std::endl;
// }

std::cout<<"\nKernel ---------- Number of blocks: "<<block_count<<"\n";

writeC:
if(block_count == 1) //// if this is the iteration where the last partial product AijBji is computed for Cmn
{
    for (int loc = 0; loc < c_row * c_col; loc++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size* c_size max = c_size * c_size
        c[loc] = C_temp[loc];   //transfer data from temp to global memory
    }

for (int loc = 0, i = 0, j = 0; loc < c_row * c_col; loc++, j++) {
        if (j == c_col) {
            i++;
            j = 0;
        }
            C_temp[loc] = 0;  /// reinitialize local variable to 0
    }

// for (int i = 0; i < FIXED_N; i++)
// {
//     for (int j = 0; j < FIXED_N; j++)
//     {
//         C_temp[i * FIXED_N + j] = 0;
//     }
// }

}

}
}