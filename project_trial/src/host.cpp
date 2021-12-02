#include "host.hpp"
#include <algorithm>
#include <cmath>
#include <torch/extension.h>
#include <torch/torch.h>

#define BLOCK_SIZE 16
#define THRESHOLD 0.0001f
#define BIN "/home/centos/project_trial/build/multadd.sw_emu.xclbin"

void fpga_matmul(std::vector<float, aligned_allocator<float>> const &source_in1_full,
                 std::vector<float, aligned_allocator<float>> const &source_in2_full,
                 std::vector<float, aligned_allocator<float>> &source_hw_results_full,
                 int DATA1_ROWS, int DATA_INNER, int DATA2_COLS, int DATA_SIZE) {
    
    static std::string binaryFile = BIN;
    static cl_int err;
    static unsigned fileBufSize;

    static std::vector<cl::Device> devices = get_devices("Xilinx");
    static bool resized = false;
    if (!resized) {
        devices.resize(1);
        resized = true;
    }
    static cl::Device device = devices[0];

    static cl::Context context(device, NULL, NULL, NULL, &err);
    static cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);

    static char *fileBuf = read_binary_file(binaryFile, fileBufSize);
    static cl::Program::Binaries bins{{fileBuf, fileBufSize}};

    static cl::Program program(context, devices, bins, NULL, &err);

    size_t matrix_size = DATA_SIZE * DATA_SIZE;
    size_t matrix_size_bytes = sizeof(float) * BLOCK_SIZE * BLOCK_SIZE;

    static cl::Kernel krnl_systolic_array(program, "multadd:{multadd_1}", &err);
    std::vector<float, aligned_allocator<float> > source_sw_results_full(matrix_size);
    // Allocate Memory for matrix block 
    std::vector<float,aligned_allocator<float>> source_in1(BLOCK_SIZE * BLOCK_SIZE);
    std::vector<float,aligned_allocator<float>> source_in2(BLOCK_SIZE * BLOCK_SIZE);
    std::vector<float,aligned_allocator<float>> source_out(BLOCK_SIZE * BLOCK_SIZE);
    std::vector<float,aligned_allocator<float>> sum(BLOCK_SIZE * BLOCK_SIZE);
    std::fill(source_in1.begin(), source_in1.end(), 0);
    std::fill(source_in2.begin(), source_in2.end(), 0);
    std::fill(source_out.begin(), source_out.end(), 0);
    std::fill(sum.begin(), sum.end(), 0);

    // Send 0's to clear kernel values ///////////////////////////////////////////////

    int block_count = DATA_SIZE / BLOCK_SIZE;  /////////Project

        for (int k = 0; k < DATA_SIZE; k += BLOCK_SIZE) 
        {

            std::cout<<"\nHost ---------- Number of blocks: "<<block_count<<"\n";

            source_in1.clear();
            source_in2.clear();
            source_out.clear();
            source_out.resize(BLOCK_SIZE * BLOCK_SIZE);
        // Copy over matrix blocks
            for (int x = 0; x < BLOCK_SIZE; x++) 
            {
                for (int y = 0; y < BLOCK_SIZE; y++) 
                {
                    source_in1.push_back(0);
                }
            }
            for (int x = 0; x < BLOCK_SIZE; x++) 
            {
                for (int y = 0; y < BLOCK_SIZE; y++) 
                {
                    source_in2.push_back(0);
                }
            }	

            // Allocate Buffer in Global Memory
            OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, matrix_size_bytes,
                                        source_in1.data(), &err));
            OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, matrix_size_bytes,
                                        source_in2.data(), &err));
            OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, matrix_size_bytes,
                                        source_out.data(), &err));

            int a_row = BLOCK_SIZE;
            int a_col = BLOCK_SIZE;
            int b_col = BLOCK_SIZE;
            

            OCL_CHECK(err, err = krnl_systolic_array.setArg(0, buffer_in1));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(1, buffer_in2));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(2, buffer_output));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(3, a_row));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(4, a_col));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(5, b_col));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(6, block_count)); /////////////Project

            // Copy input data to device global memory
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));

            // Launch the Kernel
            OCL_CHECK(err, err = q.enqueueTask(krnl_systolic_array));

            // Copy Result from Device Global Memory to Host Local Memory
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
            q.finish();
        }

    // actual block partition to send proper data to kernel
    for (int i = 0; i < DATA_SIZE; i += BLOCK_SIZE) 
    {

    for (int j = 0; j < DATA_SIZE; j += BLOCK_SIZE) 
    {
        sum.clear();
        sum.resize(BLOCK_SIZE * BLOCK_SIZE); 

        int block_count = DATA_SIZE / BLOCK_SIZE;  /////////Project

        for (int k = 0; k < DATA_SIZE; k += BLOCK_SIZE) 
        {

            std::cout<<"\nHost ---------- Number of blocks: "<<block_count<<"\n";

            source_in1.clear();
            source_in2.clear();
            source_out.clear();
            source_out.resize(BLOCK_SIZE * BLOCK_SIZE);
        // Copy over matrix blocks
            for (int x = i; x < i + BLOCK_SIZE; x++) 
            {
                for (int y = k; y < k + BLOCK_SIZE; y++) 
                {
                    source_in1.push_back(source_in1_full[x * DATA_SIZE + y]);
                }
            }
            for (int x = k; x < k + BLOCK_SIZE; x++) 
            {
                for (int y = j; y < j + BLOCK_SIZE; y++) 
                {
                    source_in2.push_back(source_in2_full[x * DATA_SIZE + y]);
                }
            }	

            // Allocate Buffer in Global Memory
            OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, matrix_size_bytes,
                                        source_in1.data(), &err));
            OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, matrix_size_bytes,
                                        source_in2.data(), &err));
            OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, matrix_size_bytes,
                                        source_out.data(), &err));

            int a_row = BLOCK_SIZE;
            int a_col = BLOCK_SIZE;
            int b_col = BLOCK_SIZE;
            

            OCL_CHECK(err, err = krnl_systolic_array.setArg(0, buffer_in1));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(1, buffer_in2));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(2, buffer_output));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(3, a_row));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(4, a_col));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(5, b_col));
            OCL_CHECK(err, err = krnl_systolic_array.setArg(6, block_count)); /////////////Project
            
            block_count--;    /////////////Project

            // Copy input data to device global memory
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));

            // Launch the Kernel
            OCL_CHECK(err, err = q.enqueueTask(krnl_systolic_array));

            // Copy Result from Device Global Memory to Host Local Memory
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
            q.finish();
            // OPENCL HOST CODE AREA END
            // Accumulate answer from block multiplication

            ////////////////////////////////////////Project//////////////////////////////////
            // for (int x = 0; x < N * N; x++) 
            // {
            // 	sum[x] += source_out[x];
            // }
            ////////////////////////////////////////Project//////////////////////////////////
        }
        
        ////////////////////////////////////////Project//////////////////////////////////
        
        auto it = source_out.begin();
        for (int x = i; x < i + BLOCK_SIZE; x++) 
        {	
            for (int y = j; y < j + BLOCK_SIZE; y++) 
            {
                source_hw_results_full[x * DATA_SIZE + y] = *it;
                ++it;
            }
        }   
    }
    }    
}

void matmul(std::vector<float, aligned_allocator<float>> &matA,
            std::vector<float, aligned_allocator<float>> &matB,
            std::vector<float, aligned_allocator<float>> &matC,
            int ARows, int ACols, int BCols) {
    for (int i = 0; i < ARows; ++i) {
        for (int j = 0; j < BCols; ++j) {
            float sum = 0;
            for (int k = 0; k < ACols; ++k) {
                sum += matA[i * ACols + k] * matB[k * BCols + j];
            }
            matC[i * BCols + j] = sum;
        }
    }
}

int blockPad(int dim) {
    return dim + (dim % BLOCK_SIZE == 0 ? 0 : (BLOCK_SIZE - dim % BLOCK_SIZE));
}

int index(int N, int C, int H, int W, int n, int c, int h, int w) {
    return n * (C * H * W) + c * (H * W) + h * (W) + w;
}

torch::Tensor fpga_conv2d(torch::Tensor input_tensor, torch::Tensor kernel_tensor, torch::Tensor bias_tensor) {
    float *images = input_tensor.data_ptr<float>();
    float *kernels = kernel_tensor.data_ptr<float>();
    float *biases = bias_tensor.data_ptr<float>();
    int N = input_tensor.sizes()[0];
    int C = input_tensor.sizes()[1];
    int H = input_tensor.sizes()[2];
    int W = input_tensor.sizes()[3];

    int K = kernel_tensor.sizes()[0];
    int R = 3;
    int S = 3;
    int stride = 1;
    int padding = 0;
    int P = ((H - R + 2 * padding) / stride) + 1;
    int Q = ((W - S + 2 * padding) / stride) + 1;

    int padded_H = H + 2 * padding;
    int padded_W = W + 2 * padding;

    std::vector<float> padded_images;
    padded_images.reserve(N * C * padded_H * padded_W);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < padded_H; ++h) {
                for (int w = 0; w < padded_W; ++w) {
                    if (h == 0 || w == 0 || h == padded_H - 1 || w == padded_W - 1) {
                        padded_images.push_back(0.f);
                    }
                    else {
                        padded_images.push_back(*images);
                        images++;
                    }
                }
            }
        }
    }

    // Flatten tensors into matrices A and B
    // A : K x C*R*S
    // B : C*R*S x N*P*Q

    // Pad matA and matB row and column wise separately to be multiples of 16
    std::vector<float, aligned_allocator<float>> matA;
    int ARows = blockPad(K);
    int ACols = blockPad(C * R * S);
    matA.resize(ARows * ACols);
    std::vector<float, aligned_allocator<float>> matB;
    int BRows = blockPad(C * R * S);
    int BCols = blockPad(N * P * Q);
    matB.resize(BRows * BCols);

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < C * R * S; ++j) {
            matA[i * ACols + j] = kernels[i * (C * R * S) + j];
        }
    }

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            int colI = c * R * S;
            int colJ = n * P * Q;
            // (i, j) selects top left corner to start convolution
            for (int i = 0; i <= padded_H - R; ++i) {
                for (int j = 0; j <= padded_W - S; ++j) {
                    for (int x = i; x < i + R; ++x) {
                        for (int y = j; y < j + S; ++y) {
                            matB[colI * BCols + colJ] = 
                                padded_images[index(N, C, padded_H, padded_W, n, c, x, y)];
                            colI++;
                            if (colI == (c + 1) * R * S) {
                                colI = c * R * S;
                                colJ++;
                            }
                        }
                    }
                }
            }
        }
    }

    // Make call to FPGA hardware
    std::vector<float, aligned_allocator<float>> matC;
    matC.resize(ARows * BCols);
    fpga_matmul(matA, matB, matC, ARows, ACols, BCols, H);
    // matmul(matA, matB, matC, ARows, ACols, BCols);
    std::vector<float> output;
    output.resize(N * K * P * Q);

    for (int n = 0; n < N; ++n) {
        int rowI = 0;
        int rowJ = n * P * Q;
        for (int k = 0; k < K; ++k) {
            float bias = biases[k];
            for (int p = 0; p < P; ++p) {
                for (int q = 0; q < Q; ++q) {
                    output[index(N, K, P, Q, n, k, p, q)] = matC[rowI * BCols + rowJ] + bias;
                    rowJ++;
                    if (rowJ == (n + 1) * P * Q) {
                        rowI++;
                        rowJ = n * P * Q;
                    }
                }
            }
        }
    }
    torch::Tensor output_tensor = torch::from_blob(output.data(), {N, K, P, Q}, input_tensor.options());
    return output_tensor.clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fpga_conv2d", &fpga_conv2d, "FPGA Conv2D");
}
