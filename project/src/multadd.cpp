/**
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
**/

/*********
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce latency and 
    device resource utilization of the resulting RTL code
    This is vector addition example to demonstrate how HLS optimizations are used in kernel. 
*********/


#include <stdio.h>
#include <iostream>
//#include <cstdlib.h>
#include <ap_fixed.h>


typedef ap_fixed<24, 10> fdx_t;
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
           int num_of_chunks    /////////////Project
           ) {

    //std::cout<<"\nKernel ---------- Number of Kernel Calls: "<<num_of_kernel_calls++<<"\n";

    #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2

    #pragma HLS INTERFACE s_axilite port=a bundle=control
    #pragma HLS INTERFACE s_axilite port=b bundle=control
    #pragma HLS INTERFACE s_axilite port=c bundle=control
    #pragma HLS INTERFACE s_axilite port=a_row bundle=control
    #pragma HLS INTERFACE s_axilite port=a_col bundle=control
    #pragma HLS INTERFACE s_axilite port=b_col bundle=control
    #pragma HLS INTERFACE s_axilite port=num_of_blocks bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int b_row = a_col;
    int c_row = a_row;
    int c_col = b_col;

    // Local memory to store input and output matrices
    fdx_t localA[FIXED_N][FIXED_N];
#pragma HLS ARRAY_PARTITION variable = localA dim = 1 complete

    fdx_t localB[FIXED_N][FIXED_N];
#pragma HLS ARRAY_PARTITION variable = localB dim = 2 complete

    fdx_t localC[FIXED_N][FIXED_N];
#pragma HLS ARRAY_PARTITION variable = localC dim = 0 complete


/////////////Project////////////////////////////////////////////////////////
    float c_temp[FIXED_N * FIXED_N];
/////////////Project////////////////////////////////////////////////////////

// cout <<" b is : "<<endl;
// for(int i =0; i<FIXED_N*FIXED_N*num_of_chunks; i++)
// {
//     cout<<b[i]<<endl;
// }

// cout <<" a is : "<<endl;
// for(int i =0; i<FIXED_N*FIXED_N*num_of_chunks; i++)
// {
//     cout<<a[i]<<endl;
// }


for(int m = 0; m < num_of_chunks; m++)
{
    // for(int n = 0; n < (FIXED_N * num_of_chunks); n += FIXED_N)
    // {
// Burst reads on input matrices from global memory
// Read Input A
// Auto-pipeline is going to apply pipeline to these loops
readA:
    for (int loc = 0, i = 0, j = 0; loc < FIXED_N * FIXED_N; loc++, j++) 
    {
        #pragma HLS LOOP_TRIPCOUNT min = c_size * c_size max = c_size * c_size
        if (j == a_col) 
        {
            i++;
            j = 0;
        }
        localA[i][j] = a[i * (FIXED_N * num_of_chunks) + (m * FIXED_N) + j];
    }


// cout<<"\nPrinting Local A in Kernel:\n";
// for(int i = 0; i < FIXED_N; i++)
// {
//     for (int j = 0; j < FIXED_N; j++)
//     {
//         cout<<localA[i][j];
//         cout<<" ";
//     }
//     cout<<endl;
// }

// Read Input B
readB:
    for (int loc = 0, i = 0, j = 0; loc < FIXED_N * FIXED_N; loc++, j++) 
    {
        #pragma HLS LOOP_TRIPCOUNT min = c_size * c_size max = c_size * c_size
        if (j == b_col) 
        {
            i++;
            j = 0;
        }
        localB[i][j] = b[i * (FIXED_N) + j + (m * FIXED_N * FIXED_N)];
    }

// cout<<"\nPrinting Local B in Kernel:\n";
// for(int i = 0; i < FIXED_N; i++)
// {
//     for (int j = 0; j < FIXED_N; j++)
//     {
//         cout<<localB[i][j];
//         cout<<" ";
//     }
//     cout<<endl;
// }

systolic1:
    for (int i = 0; i < a_col; i++) {
#pragma HLS UNROLL
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
    systolic2:
        for (int k = 0; k < FIXED_N; k++) {
#pragma HLS UNROLL
#pragma HLS PIPELINE II=1
        systolic3:
            for (int j = 0; j < FIXED_N; j++) {
#pragma HLS UNROLL
#pragma HLS PIPELINE II=1
                // Get previous sum
                fdx_t last = (k == 0) ? (fdx_t)0 : localC[i][j];

                // Update current sum
                // Handle boundary conditions
                fdx_t a_val = (i < a_row && k < a_col) ? localA[i][k] : (fdx_t)0;
                fdx_t b_val = (k < b_row && j < b_col) ? localB[k][j] : (fdx_t)0;

                fdx_t result = (last +  a_val *  b_val);
                //float result = (last + a_val * b_val);
                // Write back results

                localC[i][j] = result;
            }
        }
    }
    for (int loc = 0, i = 0, j = 0; loc < c_row * c_col; loc++, j++) {
#pragma HLS UNROLL
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = c_size* c_size max = c_size * c_size
        if (j == c_col) {
            i++;
            j = 0;
        }

        if(flag == 1)
        {
            c_temp[loc] = (float)localC[i][j];
        }
        else
            c_temp[loc] += (float)localC[i][j];
    }
    flag = 0;
}
//}

// Burst write from output matrices to global memory
// Burst write from matrix C

/////////////////////////////////Project////////////////////////////////////////////

    // cout<<"MATRIX SUM IN KERNEL\n";
    // for (int i=0;i<a_row;i++)
    // {
    //     for (int j=0;j<a_col;j++)
    //     {
    //         cout<<*(sum+i*16+j)<<" ";
    //     }
    //     cout<<"\n";
    // }

//std::cout<<"\nKernel ---------- Sum: "<<sum[0]<<"\n";

// for (int i = 0; i < FIXED_N; i++)
// {
//     for (int j = 0; j < FIXED_N; j++)
//     {
//         //std::cout<<sum[i + j * FIXED_N];
//     }
//     //std::cout<<std::endl;
// }

//std::cout<<"\nKernel ---------- Number of blocks: "<<num_of_blocks<<"\n";

writeC:
//if(num_of_blocks == 1)
//{
    for (int loc = 0; loc < c_row * c_col; loc++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size* c_size max = c_size * c_size
        c[loc] = c_temp[loc];
    }

for (int loc = 0, i = 0, j = 0; loc < c_row * c_col; loc++, j++) {
        if (j == c_col) {
            i++;
            j = 0;
        }
            c_temp[loc] = 0;
    }
//}

// for (int i = 0; i < FIXED_N; i++)
// {
//     for (int j = 0; j < FIXED_N; j++)
//     {
//         sum[i * FIXED_N + j] = 0;
//     }
// }

/////////////////////////////////Project////////////////////////////////////////////

}
}