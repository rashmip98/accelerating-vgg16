#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "host.hpp"
#include <cstdlib>
#include <cmath>


using namespace std;
using namespace torch;
namespace F = torch::nn::functional;
using namespace torch::indexing;


#define binfile "/home/centos/project//multadd.awsxclbin" 
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
  int flag = 1;
  std::vector<float, aligned_allocator<float> > source_hw_results_full(padded_kernel_2D_rowsize*padded_input_2D_colsize*2, 0); //this should be size of mat mult product = m x k
  std::vector<float, aligned_allocator<float> > source_sw_results_full(padded_kernel_2D_rowsize*padded_input_2D_colsize, 0); //this should be size of mat mult product = m x k
  // for (int i = 0; i < padded_kernel_2D_rowsize*padded_input_2D_colsize; i++) 
  //   {
  //       source_sw_results_full[i] = 0;
  //       source_hw_results_full[i] = 0;
  //   }
  int chunk_blocks = padded_kernel_2D_colsize / BLOCK_SIZE;

  size_t matrix_size_bytes = sizeof(float) * BLOCK_SIZE * BLOCK_SIZE;  
  size_t input_matrix_size_bytes = sizeof(float) *BLOCK_SIZE * BLOCK_SIZE * chunk_blocks; 

  static std::string binaryFile = binfile;
  static cl_int err;
  static unsigned fileBufSize;
    
  static std::vector<cl::Device> devices = get_devices("Xilinx");
  devices.resize(1);
  static cl::Device device = devices[0];
  //std::cout<<"got device \n";
    // ------------------------------------------------------------------------------------
    // Step 1: Create Context
    // ------------------------------------------------------------------------------------
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
	
    // ------------------------------------------------------------------------------------
    // Step 1: Create Command Queue
    // ------------------------------------------------------------------------------------
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

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
    OCL_CHECK(err, cl::Kernel krnl_systolic_array_1(program,"multadd:{multadd_1}", &err));
    OCL_CHECK(err, cl::Kernel krnl_systolic_array_2(program,"multadd:{multadd_2}", &err));
    
      clock_t start = clock();
      // Allocate Memory for matrix block 
      std::vector<float,aligned_allocator<float>> source_in1(BLOCK_SIZE * BLOCK_SIZE * chunk_blocks,0);
      std::vector<float,aligned_allocator<float>> source_in2_A(BLOCK_SIZE * BLOCK_SIZE * chunk_blocks,0);
      std::vector<float,aligned_allocator<float>> source_in2_B(BLOCK_SIZE * BLOCK_SIZE * chunk_blocks,0);
      std::vector<float,aligned_allocator<float>> source_out_A(BLOCK_SIZE * BLOCK_SIZE,0);
      std::vector<float,aligned_allocator<float>> source_out_B(BLOCK_SIZE * BLOCK_SIZE,0);

      std::vector<float,aligned_allocator<float>> odd_manager(BLOCK_SIZE * BLOCK_SIZE * chunk_blocks,0);
      
                // Allocate Buffer in Global Memory
                
                OCL_CHECK(err, cl::Buffer buffer_in1_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_matrix_size_bytes,
                                       source_in1.data(), &err));
                OCL_CHECK(err, cl::Buffer buffer_in1_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_matrix_size_bytes,
                                       source_in1.data(), &err));
                OCL_CHECK(err, cl::Buffer buffer_in2_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_matrix_size_bytes,
                                         source_in2_A.data(), &err));
                OCL_CHECK(err, cl::Buffer buffer_in2_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_matrix_size_bytes,
                                         source_in2_B.data(), &err));
                OCL_CHECK(err, cl::Buffer buffer_output_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, matrix_size_bytes,
                                            source_out_A.data(), &err));
                OCL_CHECK(err, cl::Buffer buffer_output_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, matrix_size_bytes,
                                            source_out_B.data(), &err));

                int a_row = BLOCK_SIZE;
                int a_col = BLOCK_SIZE;
                int b_col = BLOCK_SIZE;
                

                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(0, buffer_in1_A));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(1, buffer_in2_A));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(2, buffer_output_A));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(3, a_row));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(4, a_col));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(5, b_col));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(6, chunk_blocks)); // Project

                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(0, buffer_in1_B));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(1, buffer_in2_B));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(2, buffer_output_B));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(3, a_row));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(4, a_col));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(5, b_col));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(6, chunk_blocks));

                cl::Event write_event_clear;
                cl::Event task_event_A_clear;
                cl::Event task_event_B_clear;
                cl::Event read_event_clear;
                vector<cl::Event> iteration_events;

                 // Copy input data to device global memory
                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1_A, buffer_in1_B, buffer_in2_A, buffer_in2_B}, 0 /* 0 means from host*/, nullptr, &write_event_clear));
                iteration_events.push_back(write_event_clear);
                // Launch the Kernel
            
                OCL_CHECK(err, err = q.enqueueTask(krnl_systolic_array_1, &iteration_events, &task_event_A_clear));
                iteration_events.push_back(task_event_A_clear);
                OCL_CHECK(err, err = q.enqueueTask(krnl_systolic_array_2, &iteration_events, &task_event_B_clear));
                iteration_events.push_back(task_event_B_clear);

                // Copy Result from Device Global Memory to Host Local Memory
          
                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_A, buffer_output_B}, CL_MIGRATE_MEM_OBJECT_HOST, &iteration_events, &read_event_clear));
                read_event_clear.wait();
                iteration_events.push_back(read_event_clear);
                q.finish();
            
    // actual block partition to send proper data to kernel
      for (int i = 0; i < padded_kernel_2D_rowsize; i += BLOCK_SIZE) 
	  {   
        flag = 0;
        for (int j = 0; j < padded_input_2D_colsize; j += 2*BLOCK_SIZE) 
		{
	    		source_in1.clear();
	    		source_in2_A.clear();
          source_in2_B.clear();
	    		source_out_A.clear();
          source_out_B.clear();
	    		source_out_A.resize(BLOCK_SIZE*BLOCK_SIZE);
          source_out_B.resize(BLOCK_SIZE*BLOCK_SIZE);
	    	// Copy over matrix blocks
	    		for (int x = i; x < i + BLOCK_SIZE; x++) 
				{
	      			for (int y = 0; y < BLOCK_SIZE * chunk_blocks; y++) 
				  	{
						source_in1.push_back(source_in1_full[x * padded_kernel_2D_colsize + y]);
	      		  	}
	    		}
	    		for (int x = 0; x < BLOCK_SIZE * chunk_blocks; x++) 
				{
	      			for (int y = j; y < j + BLOCK_SIZE; y++) 
					{
						source_in2_A.push_back(source_in2_full[x * padded_input_2D_colsize + y]);
            
            if((y+ BLOCK_SIZE) < padded_input_2D_colsize)
            {
            source_in2_B.push_back(source_in2_full[x * padded_input_2D_colsize + y + BLOCK_SIZE]);
	      			}
            else
            {
              flag =1;
              source_in2_B.push_back(0);
            }
	    		}
        }	

                // Allocate Buffer in Global Memory
                OCL_CHECK(err, cl::Buffer buffer_in1_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_matrix_size_bytes,
                                         source_in1.data(), &err));
                OCL_CHECK(err, cl::Buffer buffer_in1_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_matrix_size_bytes,
                                         source_in1.data(), &err));
                OCL_CHECK(err, cl::Buffer buffer_in2_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_matrix_size_bytes,
                                         source_in2_A.data(), &err));
                OCL_CHECK(err, cl::Buffer buffer_in2_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_matrix_size_bytes,
                                         source_in2_B.data(), &err));
                OCL_CHECK(err, cl::Buffer buffer_output_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, matrix_size_bytes,
                                            source_out_A.data(), &err));
                OCL_CHECK(err, cl::Buffer buffer_output_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, matrix_size_bytes,
                                            source_out_B.data(), &err));

                int a_row = BLOCK_SIZE;
                int a_col = BLOCK_SIZE;
                int b_col = BLOCK_SIZE;
                

                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(0, buffer_in1_A));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(1, buffer_in2_A));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(2, buffer_output_A));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(3, a_row));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(4, a_col));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(5, b_col));
                OCL_CHECK(err, err = krnl_systolic_array_1.setArg(6, chunk_blocks));

                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(0, buffer_in1_B));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(1, buffer_in2_B));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(2, buffer_output_B));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(3, a_row));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(4, a_col));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(5, b_col));
                OCL_CHECK(err, err = krnl_systolic_array_2.setArg(6, chunk_blocks)); 
                
                cl::Event write_event1;
                cl::Event task_event_A1;
                cl::Event task_event_B1;
                cl::Event read_event1;
                // Copy input data to device global memory

                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1_A, buffer_in1_B, buffer_in2_A, buffer_in2_B}, 0 /* 0 means from host*/, &iteration_events, &write_event1));
                iteration_events.push_back(write_event1);
              
                // Launch the Kernel
                OCL_CHECK(err, err = q.enqueueTask(krnl_systolic_array_1, &iteration_events, &task_event_A1));
                iteration_events.push_back(task_event_A1);
                
                OCL_CHECK(err, err = q.enqueueTask(krnl_systolic_array_2, &iteration_events, &task_event_B1));
                iteration_events.push_back(task_event_B1);

                // Copy Result from Device Global Memory to Host Local Memory
          
                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_A, buffer_output_B}, CL_MIGRATE_MEM_OBJECT_HOST, &iteration_events, &read_event1));
                read_event1.wait();
                iteration_events.push_back(read_event1);
                //block_count--;
                q.finish();
                // OPENCL HOST CODE AREA END
               
          
            auto it_A = source_out_A.begin();
            auto it_B = source_out_B.begin();
	  	    for (int x = i; x < i + BLOCK_SIZE; x++) 
		    {	
	            for (int y = j; y < j + BLOCK_SIZE; y++) 
		        {
	      	        source_hw_results_full[x * padded_input_2D_colsize + y] = *it_A;
                  source_hw_results_full[x * padded_input_2D_colsize + y+ BLOCK_SIZE] = *it_B;
	      		    ++it_A;
                ++it_B;
	            }
	  	    }   
	    }
      }	
    source_hw_results_full.resize(padded_kernel_2D_rowsize*padded_input_2D_colsize);
    clock_t finish = clock();

    //std::cout << "Time: " << (double)(finish - start) / CLOCKS_PER_SEC << " secs" << std::endl;
    //Compute Software Results
    //expected_results(source_in1_full, source_in2_full, source_sw_results_full, padded_kernel_2D_rowsize, padded_kernel_2D_colsize, padded_input_2D_colsize);

    
    //Compare the results of the Device to the simulation
    // int match = 0;
    // for (int i = 0; i < padded_kernel_2D_rowsize*padded_input_2D_colsize; i++) {
    //     if (abs(source_hw_results_full[i] - source_sw_results_full[i]) > 3 ) 
    //     {
    //         std::cout << "Error: Result mismatch" << std::endl;
    //         std::cout << "i = " << i << " CPU result = " << source_sw_results_full[i]
    //                   << " Device result = " << source_hw_results_full[i] << std::endl;
    //         match = 1;
    //         //cout<<"final i "<<i/DATA_SIZE <<" final J "<<i%DATA_SIZE<<endl;
    //         //break;
    //     }
    // }

    //std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    // std::cout<<"Result from hardware ";
  
    // for (int i = 0; i < padded_kernel_2D_rowsize * padded_input_2D_colsize; i++) 
    //     {
    //         std::cout<<source_hw_results_full[i]<<" ";
    //         //std::cout<<std::endl;
    //     }


    return source_hw_results_full;
}


torch::Tensor convolution_fpga(Tensor input, Tensor filter, unsigned int stride, int pad) //, Tensor bias)
{
  
  int batch_N = input.size(0);          ///N
  int input_channels = input.size(1);   // C
  int input_height = input.size(2);     // H
  int input_width = input.size(3);      // W 


  int filter_size = filter.size(2);  
  int  op_channels = filter.size(0);
  
  int op_height = ((input_height - filter_size +2*pad)/stride) + 1 ;
  int op_width = ((input_width - filter_size +2*pad)/stride) + 1 ;
    
////////////////////////im2col operation
 torch::Tensor filter_flatten_temp = filter.flatten(1); 
 
auto image_split = torch::split(input, 1, 0);
torch::Tensor temp;
torch::Tensor new_img = torch::nn::functional::unfold(image_split[0], F::UnfoldFuncOptions({filter_size, filter_size}).padding(pad).stride(stride).dilation(1));
for(int i =1; i < batch_N; i++){
  temp = F::unfold(image_split[i], F::UnfoldFuncOptions({filter_size, filter_size}).padding(pad).stride(stride).dilation(1));
  new_img = torch::cat({new_img, temp}, 2);
}

/////////////////////padding
int HW = new_img.size(2);
new_img = new_img.view({new_img.size(1), new_img.size(2)});
int padded_input_row = new_img.size(0) + (new_img.size(0) % 16 == 0 ? 0 : (16 - new_img.size(0) % 16));
int padded_input_col = new_img.size(1) + (new_img.size(1) % 16 == 0 ? 0 : (16 - new_img.size(1) % 16));
int padded_kernel_row = op_channels + (op_channels % 16 == 0 ? 0 : (16 - op_channels % 16));
int padded_kernel_col = new_img.size(0) + (new_img.size(0) % 16 == 0 ? 0 : (16 - new_img.size(0) % 16));

new_img = torch::nn::functional::pad(new_img, torch::nn::functional::PadFuncOptions({0,padded_input_col - new_img.size(1),0, padded_input_row - new_img.size(0)}));
torch::Tensor new_kernel = torch::nn::functional::pad(filter_flatten_temp, torch::nn::functional::PadFuncOptions({0,padded_kernel_col - filter_flatten_temp.size(1),0,padded_kernel_row - filter_flatten_temp.size(0)}));

auto ip_new_ht = new_img.size(0);
auto ip_new_wt = new_img.size(1);
auto kernel_new_ht = new_kernel.size(0);
auto kernel_new_wt = new_kernel.size(1);



///////////////////convert  ip tensor to array

std::vector<float, aligned_allocator<float>> source_in1_full;
std::vector<float, aligned_allocator<float>> source_in2_full;

//for(int k=0; k<batch_N; k++)
//{
float* new_img_ptr = new_img.data_ptr<float>();
 for(int i = 0; i < ip_new_ht * ip_new_wt; i++)
 {
   source_in1_full.push_back(*new_img_ptr);
   new_img_ptr++;
 }

//}
 

 float* new_kernel_ptr = new_kernel.data_ptr<float>();
 for(int i = 0; i < kernel_new_ht * kernel_new_wt; i++)
 {
   source_in2_full.push_back(*new_kernel_ptr);
   new_kernel_ptr++;
 }


/////////////////call top opencl code
std::vector<float, aligned_allocator<float>> source_hw_results_full = host_to_kernel(source_in2_full, source_in1_full, kernel_new_ht, kernel_new_wt, ip_new_wt);


///////////////////////////// convert vector to tensor
torch::Tensor torch_product = torch::from_blob(source_hw_results_full.data(), {kernel_new_ht, ip_new_wt});
//std::cout<<"Torch dimensions "<<torch_product.dim();

///////////////////////////// de-pad
torch::Tensor torch_result = torch_product.index({Slice(None, op_channels), Slice(None, HW)}); // N, HW as per manas
////std::cout<<"Torch product "<<torch_result<<std::endl;

////////////////////////////Fold de-pad output
auto splitOutput = torch::split(torch_result, (HW/batch_N), 1);
torch::Tensor tempOut = splitOutput[0];
for(int i =1; i < batch_N; i++){
  tempOut = torch::cat({tempOut, splitOutput[i]}, 0);
}
torch::Tensor torch_reshape_result = tempOut.view({batch_N, op_channels, op_height, op_width}) ;

return torch_reshape_result.clone();
}

PYBIND11_MODULE(host, m) 
{
  m.def("convolution_fpga", &convolution_fpga,"convolution fpga small trial");
}