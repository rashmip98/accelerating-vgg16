#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "host.hpp"
#include <cstdlib>
#include <cmath>

//#include <fold.h>

using namespace std;
using namespace torch;
namespace F = torch::nn::functional;

#define binfile "/home/centos/q4/build/multadd.sw_emu.xclbin" 

void expected_results(std::vector<float, aligned_allocator<float>>& in1, // Input Matrix 1
                    std::vector<float, aligned_allocator<float>>& in2, // Input Matrix 2
                    std::vector<float, aligned_allocator<float>>& out, int DATA_SIZE  // Output Matrix
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

std::vector<float, aligned_allocator<float>> host_to_kernel(std::vector<float, aligned_allocator<float>>& source_in1_full, 
                    std::vector<float, aligned_allocator<float>>& source_in2_full, 
                    unsigned int input_size, //this should be rectangular m x n
                    unsigned int kernel_size) // this should also be rectangular n x k
{
  int DATA_SIZE = 32;
  int N=16;   
  std::vector<float, aligned_allocator<float> > source_hw_results_full(input_size); //this should be size of mat mult product = m x k
  std::vector<float, aligned_allocator<float> > source_sw_results_full(input_size); //this should be size of mat mult product = m x k
  for (size_t i = 0; i < input_size; i++) 
    {
        source_sw_results_full[i] = 0;
        source_hw_results_full[i] = 0;
    }
    
  // cl_int err;
  // cl::CommandQueue q;
  // cl::Context context;
  // cl::Kernel krnl_systolic_array;

  // std::string binaryFile = binfile;

  //size_t matrix_size = DATA_SIZE * DATA_SIZE;  //change for rectangular ip, kernel
  //size_t matrix_size_bytes = sizeof(float) * N * N; //change for rectangular ip, kernel

  static size_t matrix_size = DATA_SIZE*DATA_SIZE;
  static std::string binaryFile = binfile;
  static size_t matrix_size_bytes = sizeof(float) * N * N;
  static cl_int err;
  static unsigned fileBufSize;
    //simulation_results(source_in1, source_in2, source_sw_results, DATA_SIZE);
  static std::vector<cl::Device> devices = get_devices("Xilinx");
  devices.resize(1);
  static cl::Device device = devices[0];
  std::cout<<"got device \n";
    // ------------------------------------------------------------------------------------
    // Step 1: Create Context
    // ------------------------------------------------------------------------------------
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
	
    // ------------------------------------------------------------------------------------
    // Step 1: Create Command Queue
    // ------------------------------------------------------------------------------------
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    // ------------------------------------------------------------------
    // Step 1: Load Binary File from disk
    // ------------------------------------------------------------------		
    char* fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};

   // -------------------------------------------------------------
   // Step 1: Create the program object from the binary and program the FPGA device with it
   // -------------------------------------------------------------	
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

  // -------------------------------------------------------------
  // Step 1: Create Kernels
  // -------------------------------------------------------------
    OCL_CHECK(err, cl::Kernel krnl_systolic_array(program,"multadd", &err));
  // auto devices = xcl::get_xil_devices();

  //   auto fileBuf = xcl::read_binary_file(binaryFile);
  //   cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  //   bool valid_device = false;
  //   for (unsigned int i = 0; i < devices.size(); i++) {
  //       auto device = devices[i];
  //       // Creating Context and Command Queue for selected Device
  //       OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
  //       OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

  //       std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
  //       cl::Program program(context, {device}, bins, nullptr, &err);
  //       if (err != CL_SUCCESS) {
  //           std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
  //       } else {
  //           std::cout << "Device[" << i << "]: program successful!\n";
  //           OCL_CHECK(err, krnl_systolic_array = cl::Kernel(program, "multadd", &err));
  //           valid_device = true;
  //           break; // we break because we found a valid device
  //       }
  //   }
  //   if (!valid_device) {
  //       std::cout << "Failed to program any device found, exit!\n";
  //       exit(EXIT_FAILURE);
  //   }
    
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
      //int block_count = 2;

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
                OCL_CHECK(err, err = krnl_systolic_array.setArg(6, block_count)); // Project

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
    return source_hw_results_full;
}


torch::Tensor convolution_fpga(Tensor input, Tensor filter, unsigned int stride, int padding)
{
  
   auto ip_dim = input.dim(); // dimensions of the tensor
   // assuming max input tensor is 4, N * I/P_CH *width * height
   unsigned int batch_N, input_channels, input_width, input_height =0;
   //if(ip_dim==4)
  // {
     //cout<<"4"<<endl;
     batch_N = input.size(0);          ///N
     input_channels = input.size(1);   // C
     input_height = input.size(2);     // H
     input_width = input.size(3);      // W
     //cout<<"end"<<endl;
  // }
   
  //  else if(ip_dim==3)
  //  {
  //    //cout<<"3"<<endl;
  //    batch_N = 1;
  //    input_channels = input.size(0);
  //    input_height = input.size(1);
  //    input_width = input.size(2);
  //    //cout<<"end"<<endl;
  //  }

  // else if(ip_dim==2)
  //  {
  //    //cout<<"2"<<endl;
  //    batch_N = 1;
  //    input_channels = 1;
  //    input_height = input.size(0);
  //    input_width = input.size(1);
  //    //cout<<"end"<<endl;
  //  }

   // assuming filter is the square kernel 

    std::cout<<"Input tensor from python"<<input<<std::endl;
    std::cout<<"Kernel tensor from python"<<filter<<std::endl;

    unsigned int dim_filter;
    dim_filter = filter.dim();
    auto filter_size = filter.size(2);  //or 3, 4 dim tensor, 0 being output channels, 1 being ip channels and rest being kernel size
    unsigned int op_channels =0; // kernel op_channel *ip*channels *kernel size
    op_channels = filter.size(0);
    int op_height, op_width,h,w =0;
    op_height = ((input_height - filter_size +2*padding)/stride) + 1 ;
    op_width = ((input_width - filter_size +2*padding)/stride) + 1 ;

    
    cout<<"\nFilter size is : "<<filter_size;
    cout<<"\nInput height is : "<<input_height;
    cout<<"\nInput width is : "<<input_width<<endl;
   
  
    
// converting 4D tensor to 2D tensor
//filter is (Y,C,K,K) into (Y, K*K*C)
 torch::Tensor filter_flatten_temp = torch::zeros({batch_N,op_channels,filter_size*filter_size});
 filter_flatten_temp = filter.flatten(1); //flatten across first dimension, filter wont be affected
 
// need to convert image (N,C,H,W) to (N,K*K*C,H*W) BATCH SIZE NOT AFFECTED
torch::Tensor new_img = torch::nn::functional::unfold(input, F::UnfoldFuncOptions({filter_size, filter_size}).padding(0).stride(1).dilation(1));
// new_img = new_img.transpose(1,2);
//unfold-> transpose -> conv to array ->multiply -> conv to tensor -> fold 
std::cout<<"New unfolded image"<<new_img;
// auto ip_new_ht = input_height + (input_height % 16 == 0 ? 0 : (16 - dim % 16));
// auto ip_new_wt = input_width + (input_width % 16 == 0 ? 0 : (16 - dim % 16));
new_img = torch::nn::functional::pad(new_img, torch::nn::functional::PadFuncOptions({0,28,0,5}));
std::cout<<"After padding"<<new_img;

//fold
//torch::Tensor folded_tensor = torch::nn::functional::fold(new_img, F::FoldFuncOptions({4,4},{3,3}).padding(0).stride(1).dilation(1));

batch_N = new_img.size(0);
auto ip_new_ht = new_img.size(1);
auto ip_new_wt = new_img.size(2);

//convert  ip tensor to array
unsigned int matrix_size = ip_new_ht*ip_new_wt;
//unsigned int kernel_size = filter_size*filter_size;
//unsigned int kernel_size = filter_size*filter_size*op_channels*input_channels;
unsigned int kernel_size = filter_size*filter_size*input_channels;
//std::vector<float, aligned_allocator<float>> source_in1_M(matrix_size);
//std::vector<float, aligned_allocator<float>> source_in2_M(kernel_size);

std::vector<vector<float, aligned_allocator<float>>> source_in1_M( ip_new_ht , vector<float, aligned_allocator<float>> (ip_new_wt, 0));
std::vector<vector<float, aligned_allocator<float>>> source_in2_M( batch_N , vector<float, aligned_allocator<float>> (kernel_size, 0));

//input tensor
for (int k=0; k<batch_N; k++)
  {
    float* temp_arr = new_img[k].data_ptr<float>();
    cout<<"batch is : "<<k+1<<endl<<endl;
    for (int i=0; i < ip_new_ht; i++)
      {
        for (int j = 0; j < ip_new_wt; j++)
        {
          /* code */
          source_in1_M[j][i]= *temp_arr;
          temp_arr++;
          //cout<<source_in1_M[i][j]<<endl;
        }
      } 
  }
  // source_in1_M.resize(16);
  // for (int i = 0; i < source_in1_M.size(); ++i)
  //   source_in1_M[i].resize(32);

  std::cout<<"Resized input vector"<<source_in1_M<<std::endl;
  //kernel
  //for (int k=0; k<batch_N; k++)
  //{
    float* temp_arr = filter_flatten_temp.data_ptr<float>();
    //cout<<"batch is : "<<k+1<<endl<<endl;
    //for (int i=0; i < filter_size; i++)
    for (int i = 0; i < batch_N; i ++)  
      {
        for (int j = 0; j < kernel_size; j++)
        {
          /* code */
          source_in2_M[i][j]= *temp_arr;
          temp_arr++;
          //cout<<source_in2_M[i][j]<<endl;
        }
      } 
  //}
   source_in2_M.resize(32);
   for (int i = 0; i < source_in2_M.size(); ++i)
    source_in2_M[i].resize(32);

  std::cout<<"Resized filter vector"<<source_in2_M<<std::endl;

  // for (int k=0; k<batch_N; k++)
  // {
  //   float* temp_arr = new_img[k].data_ptr<float>();
  //   cout<<"batch is : "<<k+1<<endl<<endl;
  //      for (int x=0; x<matrix_size; x++)
  //     {
  //         //cout<<(*(temp_arr+x))<<endl;
  //         source_in1_M[x]= *(temp_arr+x);
  //         cout<<source_in1_M[x]<<endl;
  //     } 
  // }

//convert kernel tensor to array

// std::vector<float, aligned_allocator<float>> source_in2_M(kernel_size);

//     float* temp_arr = filter_flatten_temp.data_ptr<float>();
//     for (int x=0; x<kernel_size; x++)
//       {
//           //cout<<(*(temp_arr+x))<<endl;
//           source_in2_M[x]= *(temp_arr+x);
//           cout<<source_in2_M[x]<<endl;
//       } 
  

//multiply
// for (int i = 0; i < ip_new_ht; i++) {
//         for (int j = 0; j < input_channels*filter_size*filter_size; j++) {
//             for (int k = 0; k < ip_new_wt; k++) {
//                 out[i * DATA_SIZE + j] += in1[i * DATA_SIZE + k] * source_in2_M[k * DATA_SIZE + j];
//             }
//         }
//     }
auto padded_input_2D_colsize = source_in1_M.size();
auto padded_input_2D_rowsize = source_in1_M[0].size();
auto padded_kernel_2D_colsize = source_in2_M.size();
auto padded_kernel_2D_rowsize = source_in2_M[0].size();

std::cout<<"No. of cols in ip"<<padded_input_2D_colsize <<std::endl; //col is source_in1_M.size(), row is source_in1_M[0].size()
std::cout<<"No. of rows in ip"<<padded_input_2D_rowsize <<std::endl;
std::cout<<"No. of cols in kernel"<<padded_kernel_2D_colsize<<std::endl;
std::cout<<"No. of rows in kernel"<<padded_kernel_2D_rowsize <<std::endl;

std::vector<float, aligned_allocator<float>> source_in1_full(padded_input_2D_rowsize * padded_input_2D_colsize);
std::vector<float, aligned_allocator<float>> source_in2_full(padded_kernel_2D_rowsize * padded_kernel_2D_colsize);


/////////////////conv to one d vector
unsigned int k=0;
for (int i=0; i < padded_input_2D_rowsize; i++)
{
  for(int j=0; j < padded_input_2D_colsize; j++)
  { 
    source_in1_full[k] = source_in1_M[i][j];
    k++;
    
  }
}
unsigned int p=0;
for (int i=0; i < padded_kernel_2D_rowsize; i++)
{
  for(int j=0; j < padded_kernel_2D_colsize; j++)
  { 
    source_in2_full[p] = source_in2_M[i][j];
    p++;
    
  }
}
unsigned int input_final_size = padded_input_2D_rowsize * padded_input_2D_colsize;
unsigned int kernel_final_size = padded_input_2D_rowsize * padded_input_2D_colsize;
std::cout<<"1D padded input vector"<<source_in1_full<<std::endl;
std::cout<<"1D padded kernel vector"<<source_in2_full<<std::endl;
host_to_kernel(source_in1_full, source_in2_full, input_final_size, kernel_final_size);
return filter_flatten_temp;
}


///////////////////////////////////////////////////////////////////////////////////// host for kernel call



//std::vector<float, aligned_allocator<float>> host_blockmult(int argc, char** argv) {   
//}

PYBIND11_MODULE(host_trial, m) 
{
  m.def("convolution_fpga", &convolution_fpga,"convolution fpga small trial");
}