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
using namespace torch::indexing;


#define binfile "/home/centos/q4/build/multadd.sw_emu.xclbin" 
#define BLOCK_SIZE 16

void expected_results(std::vector<float, aligned_allocator<float>>& in1, // Input Matrix 1
                    std::vector<float, aligned_allocator<float>>& in2, // Input Matrix 2
                    std::vector<float, aligned_allocator<float>>& out, int padded_kernel_2D_rowsize, int padded_kernel_2D_colsize, int padded_input_2D_colsize  // Output Matrix
                    ) {
    // Perform Matrix multiply Out = In1 x In2
    
    for (int i = 0; i < padded_kernel_2D_rowsize; i++) {
        for (int j = 0; j < padded_input_2D_colsize; j++) {
            for (int k = 0; k < padded_kernel_2D_colsize; k++) {
                out[i * padded_input_2D_colsize + j] += in1[i * padded_kernel_2D_colsize + k] * in2[k * padded_input_2D_colsize + j];
            }
        }
    }
}

std::vector<float, aligned_allocator<float>> host_to_kernel(std::vector<float, aligned_allocator<float>>& source_in1_full, 
                    std::vector<float, aligned_allocator<float>>& source_in2_full, 
                    int padded_kernel_2D_rowsize, int padded_kernel_2D_colsize, int padded_input_2D_colsize) 
{

  std::vector<float, aligned_allocator<float> > source_hw_results_full(padded_kernel_2D_rowsize*padded_input_2D_colsize); //this should be size of mat mult product = m x k
  std::vector<float, aligned_allocator<float> > source_sw_results_full(padded_kernel_2D_rowsize*padded_input_2D_colsize); //this should be size of mat mult product = m x k
  for (int i = 0; i < padded_kernel_2D_rowsize*padded_input_2D_colsize; i++) 
    {
        source_sw_results_full[i] = 0;
        source_hw_results_full[i] = 0;
    }

  size_t matrix_size = BLOCK_SIZE * BLOCK_SIZE;  
  size_t matrix_size_bytes = sizeof(float) *BLOCK_SIZE * BLOCK_SIZE; 

  static std::string binaryFile = binfile;
  static cl_int err;
  static unsigned fileBufSize;
    
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
    
      clock_t start = clock();
      // Allocate Memory for matrix block 
      std::vector<float,aligned_allocator<float>> source_in1(matrix_size);
      std::vector<float,aligned_allocator<float>> source_in2(matrix_size);
      std::vector<float,aligned_allocator<float>> source_out(matrix_size);
      std::vector<float,aligned_allocator<float>> sum(matrix_size);
      std::fill(source_in1.begin(), source_in1.end(), 0);
      std::fill(source_in2.begin(), source_in2.end(), 0);
      std::fill(source_out.begin(), source_out.end(), 0);
      std::fill(sum.begin(), sum.end(), 0);

    // Send 0's to clear kernel values ///////////////////////////////////////////////

      int block_count = 1;  

                //std::cout<<"\nHost ---------- Number of blocks: "<<block_count<<"\n";

	    		source_in1.clear();
	    		source_in2.clear();
	    		source_out.clear();
	    		source_out.resize(matrix_size);
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
                OCL_CHECK(err, err = krnl_systolic_array.setArg(6, block_count)); // Project

                // Copy input data to device global memory
                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));

                // Launch the Kernel
                OCL_CHECK(err, err = q.enqueueTask(krnl_systolic_array));

                // Copy Result from Device Global Memory to Host Local Memory
                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
                q.finish();
            
    // actual block partition to send proper data to kernel
      for (int i = 0; i < padded_kernel_2D_rowsize; i += BLOCK_SIZE) 
	  {

        for (int j = 0; j < padded_input_2D_colsize; j += BLOCK_SIZE) 
		{
	  		sum.clear();
	  		sum.resize(BLOCK_SIZE*BLOCK_SIZE); 

       int block_count = padded_kernel_2D_colsize / BLOCK_SIZE;  /////////Project

	  		for (int k = 0; k < padded_kernel_2D_colsize; k += BLOCK_SIZE) 
			{

                std::cout<<"\nHost ---------- Number of blocks: "<<block_count<<"\n";

	    		source_in1.clear();
	    		source_in2.clear();
	    		source_out.clear();
	    		source_out.resize(BLOCK_SIZE*BLOCK_SIZE);
	    	// Copy over matrix blocks
	    		for (int x = i; x < i + BLOCK_SIZE; x++) 
				{
	      			for (int y = k; y < k + BLOCK_SIZE; y++) 
				  	{
						source_in1.push_back(source_in1_full[x * padded_kernel_2D_colsize + y]);
	      		  	}
	    		}
	    		for (int x = k; x < k + BLOCK_SIZE; x++) 
				{
	      			for (int y = j; y < j + BLOCK_SIZE; y++) 
					{
						source_in2.push_back(source_in2_full[x * padded_input_2D_colsize + y]);
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
                OCL_CHECK(err, err = krnl_systolic_array.setArg(6, block_count)); 
                

                // Copy input data to device global memory
                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));

                // Launch the Kernel
                OCL_CHECK(err, err = q.enqueueTask(krnl_systolic_array));

                // Copy Result from Device Global Memory to Host Local Memory
                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
                block_count--;
                q.finish();
                // OPENCL HOST CODE AREA END
               
	  		}
                        
            auto it = source_out.begin();
	  	    for (int x = i; x < i + BLOCK_SIZE; x++) 
		    {	
	            for (int y = j; y < j + BLOCK_SIZE; y++) 
		        {
	      	        source_hw_results_full[x * padded_input_2D_colsize + y] = *it;
	      		    ++it;
	            }
	  	    }   
	    }
      }	

    clock_t finish = clock();

    std::cout << "Time: " << (double)(finish - start) / CLOCKS_PER_SEC << " secs" << std::endl;
    // Compute Software Results
    expected_results(source_in1_full, source_in2_full, source_sw_results_full, padded_kernel_2D_rowsize, padded_kernel_2D_colsize, padded_input_2D_colsize);

    
    // Compare the results of the Device to the simulation
    int match = 0;
    for (int i = 0; i < padded_kernel_2D_rowsize*padded_input_2D_colsize; i++) {
        if (abs(source_hw_results_full[i] - source_sw_results_full[i]) > 3 ) 
        {
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_results_full[i]
                      << " Device result = " << source_hw_results_full[i] << std::endl;
            match = 1;
            //cout<<"final i "<<i/DATA_SIZE <<" final J "<<i%DATA_SIZE<<endl;
            //break;
        }
    }

    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    std::cout<<"Result from hardware ";
    //std::cout<<"Size is"<<result_from_hw.dim();
    for (int i = 0; i < padded_kernel_2D_rowsize * padded_input_2D_colsize; i++) 
        {
            std::cout<<source_hw_results_full[i]<<" ";
            //std::cout<<std::endl;
        }
    return source_hw_results_full;
}


torch::Tensor convolution_fpga(Tensor input, Tensor filter, unsigned int stride, int pad)
{
  
  // assuming max input tensor is 4, N * I/P_CH *width * height
  int batch_N = input.size(0);          ///N
  int input_channels = input.size(1);   // C
  int input_height = input.size(2);     // H
  int input_width = input.size(3);      // W 

  std::cout<<"Input tensor from python"<<input<<std::endl;
  std::cout<<"Kernel tensor from python"<<filter<<std::endl;

  int filter_size = filter.size(2);  //or 3, 4 dim tensor, 0 being output channels, 1 being ip channels and rest being kernel size
  int  op_channels = filter.size(0);
  
  int op_height = ((input_height - filter_size +2*pad)/stride) + 1 ;
  int op_width = ((input_width - filter_size +2*pad)/stride) + 1 ;

  std::cout<<"\nFilter size is : "<<filter_size;
  std::cout<<"\nInput height is : "<<input_height;
  std::cout<<"\nInput width is : "<<input_width<<endl;
   
  
    
// converting 4D tensor to 2D tensor
//filter is (Y,C,K,K) into (Y, K*K*C)
 torch::Tensor filter_flatten_temp = torch::zeros({batch_N,op_channels,filter_size*filter_size});
 filter_flatten_temp = filter.flatten(1); //flatten across first dimension, filter wont be affected
 
// need to convert image (N,C,H,W) to (N,K*K*C,H*W) BATCH SIZE NOT AFFECTED
auto inputSplit = torch::split(input, 1, 0);
torch::Tensor temp;
torch::Tensor new_img = torch::nn::functional::unfold(inputSplit[0], F::UnfoldFuncOptions({filter_size, filter_size}).padding(pad).stride(stride).dilation(1));
for(int i =1; i < batch_N; i++){
  temp = F::unfold(inputSplit[i], F::UnfoldFuncOptions({filter_size, filter_size}).padding(pad).stride(stride).dilation(1));
  new_img = torch::cat({new_img, temp}, 2);
}
// new_img = new_img.transpose(1,2);
//unfold-> transpose -> conv to array ->multiply -> conv to tensor -> fold 
std::cout<<"New unfolded image"<<new_img;
int HW = new_img.size(2);
new_img = new_img.view({new_img.size(1), new_img.size(2)});
int padded_input_row = new_img.size(0) + (new_img.size(0) % 16 == 0 ? 0 : (16 - new_img.size(0) % 16));
int padded_input_col = new_img.size(1) + (new_img.size(1) % 16 == 0 ? 0 : (16 - new_img.size(1) % 16));
int padded_kernel_row = op_channels + (op_channels % 16 == 0 ? 0 : (16 - op_channels % 16));
int padded_kernel_col = new_img.size(0) + (new_img.size(0) % 16 == 0 ? 0 : (16 - new_img.size(0) % 16));
//int pad_max = std::max(std::max(padded_input_row, padded_input_col), std::max(padded_kernel_row, padded_kernel_col));
new_img = torch::nn::functional::pad(new_img, torch::nn::functional::PadFuncOptions({0,padded_input_col - new_img.size(1),0, padded_input_row - new_img.size(0)}));
torch::Tensor new_kernel = torch::nn::functional::pad(filter_flatten_temp, torch::nn::functional::PadFuncOptions({0,padded_kernel_col - filter_flatten_temp.size(1),0,padded_kernel_row - filter_flatten_temp.size(0)}));


//batch_N = new_img.size(0);
auto ip_new_ht = new_img.size(0);
auto ip_new_wt = new_img.size(1);

auto kernel_new_ht = new_kernel.size(0);
auto kernel_new_wt = new_kernel.size(1);


//convert  ip tensor to array

std::vector<float, aligned_allocator<float>> source_in1_full;
std::vector<float, aligned_allocator<float>> source_in2_full;

for(int k=0; k<batch_N; k++)
{
float* new_img_ptr = new_img[k].data_ptr<float>();
 for(int i = 0; i < ip_new_ht * ip_new_wt; i++)
 {
   source_in1_full.push_back(*new_img_ptr);
   new_img_ptr++;
 }

}
 

 float* new_kernel_ptr = new_kernel.data_ptr<float>();
 for(int i = 0; i < kernel_new_ht * kernel_new_wt; i++)
 {
   source_in2_full.push_back(*new_kernel_ptr);
   new_kernel_ptr++;
 }


std::cout<<"1D padded input vector"<<source_in1_full<<std::endl;
std::cout<<"1D padded kernel vector"<<source_in2_full<<std::endl;

std::vector<float, aligned_allocator<float>> source_hw_results_full = host_to_kernel(source_in2_full, source_in1_full, kernel_new_ht, kernel_new_wt, ip_new_wt);


///////////////////////////// convert vector to tensor
torch::Tensor torch_product = torch::from_blob(source_hw_results_full.data(), {kernel_new_ht, ip_new_wt});
//std::cout<<"Torch dimensions "<<torch_product;

///////////////////////////// de-pad
torch::Tensor torch_result = torch_product.index({Slice(None, batch_N), Slice(None, HW)}); // batches, HW 
std::cout<<"Torch product "<<torch_result<<std::endl;

//Fold de-pad output
//torch::Tensor torch_reshape_result = torch::reshape(torch_result.data(), {batch_N, op_channels, op_height, op_width});
auto splitOutput = torch::split(torch_result, (HW/batch_N), 1);
torch::Tensor tempOut = splitOutput[0];
for(int i =1; i < batch_N; i++){
  tempOut = torch::cat({tempOut, splitOutput[i]}, 0);
}
torch::Tensor torch_reshape_result = tempOut.view({batch_N, op_channels, op_height, op_width});
std::cout<<"Torch product Reshaped "<<torch_reshape_result<<std::endl;

return torch_reshape_result.clone();
}

PYBIND11_MODULE(host_trial, m) 
{
  m.def("convolution_fpga", &convolution_fpga,"convolution fpga small trial");
}