#include "host.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

// Array Size to access
//#define DATA_SIZE 16
#define N 16

// Maximum Array Size
#define MAX_SIZE 2048

using namespace std;

// Software implementation of Matrix Multiplication
// The inputs are of the size (DATA_SIZE x DATA_SIZE)
void expected_results(std::vector<float, aligned_allocator<float> >& in1, // Input Matrix 1
                    std::vector<float, aligned_allocator<float> >& in2, // Input Matrix 2
                    std::vector<float, aligned_allocator<float> >& out, int DATA_SIZE  // Output Matrix
                    ) {
    // Perform Matrix multiply Out = In1 x In2
    
    for (int i = 0; i < DATA_SIZE; i++) {
        for (int j = 0; j < DATA_SIZE; j++) {
            for (int k = 0; k < DATA_SIZE; k++) {
                out[i * DATA_SIZE + j] += in1[i * DATA_SIZE + k] * in2[k * DATA_SIZE + j];
            }
        }
    }
}

void display(std::vector<float,aligned_allocator<float>> &mat, int width, std::string name)
{
	std::cout<<"\n\nThe matrix is: "<< name << std::endl;
	for (int i=0; i < width; i++)
	{
		std::cout<< std::endl;
		for (int j=0; j < width; j++)
			std::cout<<mat[i*width + j]<<" ";
	}
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    int DATA_SIZE = atoi(argv[2]);

    // Allocate Memory in Host Memory
    if (DATA_SIZE > MAX_SIZE) {
        std::cout << "Size is bigger than internal buffer size, please use a "
                     "size smaller than "
                  << MAX_SIZE << "!" << std::endl;
        return EXIT_FAILURE;
    }

    size_t matrix_size = DATA_SIZE * DATA_SIZE;
    size_t matrix_size_bytes = sizeof(float) * N * N;
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Kernel krnl_systolic_array;

    /////////////////////////////////////////////////// im2col //////////////////////////////////////////////////////////
    
    std::vector<float, aligned_allocator<float> > source_in1_full(matrix_size);
    std::vector<float, aligned_allocator<float> > source_in2_full(matrix_size);
    std::vector<float, aligned_allocator<float> > source_hw_results_full(matrix_size);
    std::vector<float, aligned_allocator<float> > source_sw_results_full(matrix_size);

                         /*   int in_batches=2;
                            int in_channels=2;
                            int in_height=3;
                            int in_width=3;

                            int kernel_height=2;
                            int kernel_width=2;
                            int stride=1;
                            int pad=0;

                            float input_tensor[in_batches*in_channels*in_height*in_width];
                            std::cout<<"Input:";
                            init_tensor(input_tensor, in_batches, in_channels, in_height, in_width);
                            print_tensor(input_tensor, in_batches, in_channels, in_height, in_width);

                            int out_height;
                            int out_width;

                            float * output_tensor = img2col(input_tensor, in_batches, in_channels, in_height, in_width, 
                                    kernel_height, kernel_width, stride, pad, 
                                    &out_height, &out_width);

                            std::cout<<"Output:";
                            print_tensor(output_tensor, 1, 1, out_height, out_width);
                            */
                            // Remember to delete memory
                            //delete[] output_tensor;

    // Create the test data and Software Result
    for (size_t i = 0; i < matrix_size; i++) 
    {
        source_in1_full[i] = i % 10;
        source_in2_full[i] = i % 10;
        source_sw_results_full[i] = 0;
        source_hw_results_full[i] = 0;
    }
    
    /////////////////////////////////////////////////// im2col //////////////////////////////////////////////////////////

    //display(source_in1_full,DATA_SIZE,"source_in1_full");
    //display(source_in2_full,DATA_SIZE,"source_in2_full");

    // OPENCL HOST CODE AREA START
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_systolic_array = cl::Kernel(program, "multadd", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    
      clock_t start = clock();
      // Allocate Memory for matrix block 
      std::vector<float,aligned_allocator<float>> source_in1(N * N);
      std::vector<float,aligned_allocator<float>> source_in2(N * N);
      std::vector<float,aligned_allocator<float>> source_out(N * N);
      std::vector<float,aligned_allocator<float>> sum(N * N);
      std::fill(source_in1.begin(), source_in1.end(), 0);
      std::fill(source_in2.begin(), source_in2.end(), 0);
      std::fill(source_out.begin(), source_out.end(), 0);
      std::fill(sum.begin(), sum.end(), 0);

    // Send 0's to clear kernel values ///////////////////////////////////////////////

      int block_count = DATA_SIZE / N;  

	  		for (int k = 0; k < DATA_SIZE; k += N)  /// CLEAR KERNEL WITH 0 VALUES
			{

                std::cout<<"\nHost ---------- Number of blocks: "<<block_count<<"\n";

	    		source_in1.clear();
	    		source_in2.clear();
	    		source_out.clear();
	    		source_out.resize(N * N);
	    	// Copy over matrix blocks
	    		for (int x = 0; x < N; x++) 
				{
	      			for (int y = 0; y < N; y++) 
				  	{
						source_in1.push_back(0);
	      		  	}
	    		}
	    		for (int x = 0; x < N; x++) 
				{
	      			for (int y = 0; y < N; y++) 
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

                int a_row = N;
                int a_col = N;
                int b_col = N;
                

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
      for (int i = 0; i < DATA_SIZE; i += N) 
	  {

        for (int j = 0; j < DATA_SIZE; j += N) 
		{
	  		sum.clear();
	  		sum.resize(N * N); 

            int block_count = DATA_SIZE / N;  /////////Project

	  		for (int k = 0; k < DATA_SIZE; k += N) 
			{

                std::cout<<"\nHost ---------- Number of blocks: "<<block_count<<"\n";

	    		source_in1.clear();
	    		source_in2.clear();
	    		source_out.clear();
	    		source_out.resize(N * N);
	    	// Copy over matrix blocks
	    		for (int x = i; x < i + N; x++) 
				{
	      			for (int y = k; y < k + N; y++) 
				  	{
						source_in1.push_back(source_in1_full[x * DATA_SIZE + y]);
	      		  	}
	    		}
	    		for (int x = k; x < k + N; x++) 
				{
	      			for (int y = j; y < j + N; y++) 
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

                int a_row = N;
                int a_col = N;
                int b_col = N;
                

                OCL_CHECK(err, err = krnl_systolic_array.setArg(0, buffer_in1));
                OCL_CHECK(err, err = krnl_systolic_array.setArg(1, buffer_in2));
                OCL_CHECK(err, err = krnl_systolic_array.setArg(2, buffer_output));
                OCL_CHECK(err, err = krnl_systolic_array.setArg(3, a_row));
                OCL_CHECK(err, err = krnl_systolic_array.setArg(4, a_col));
                OCL_CHECK(err, err = krnl_systolic_array.setArg(5, b_col));
                OCL_CHECK(err, err = krnl_systolic_array.setArg(6, block_count)); /////////////Project//
                
                block_count--;    /////////////Project////

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
	  	    for (int x = i; x < i + N; x++) 
		    {	
	            for (int y = j; y < j + N; y++) 
		        {
	      	        source_hw_results_full[x * DATA_SIZE + y] = *it;
	      		    ++it;
	            }
	  	    }   
	    }
      }	

      clock_t finish = clock();

    std::cout << "For DATA_SIZE= " << DATA_SIZE << ": " << (double)(finish - start) / CLOCKS_PER_SEC << " secs" << std::endl;
    // Compute Software Results
    expected_results(source_in1_full, source_in2_full, source_sw_results_full, DATA_SIZE);

    //display(source_sw_results_full,DATA_SIZE,"sw_results_M");
    //display(source_hw_results_full,DATA_SIZE,"hw_results_M");
    
    // Compare the results of the Device to the simulation
    int match = 0;
    for (int i = 0; i < DATA_SIZE * DATA_SIZE; i++) {
        if (abs(source_hw_results_full[i] - source_sw_results_full[i]) > 3 ) 
        {
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_results_full[i]
                      << " Device result = " << source_hw_results_full[i] << std::endl;
            match = 1;
            cout<<"final i "<<i/DATA_SIZE <<" final J "<<i%DATA_SIZE<<endl;
            //break;
        }
    }

    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE : EXIT_SUCCESS);
}