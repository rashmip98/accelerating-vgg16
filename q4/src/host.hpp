#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

//OCL_CHECK doesn't work if call has templatized function call
#define OCL_CHECK(error,call)                                       \
    call;                                                           \
    if (error != CL_SUCCESS) {                                      \
      printf("%s:%d Error calling " #call ", error code is: %d\n",  \
              __FILE__,__LINE__, error);                            \
      exit(EXIT_FAILURE);                                           \
    }

#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>
#include <limits.h>
#include <sys/stat.h>


template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};

namespace xcl {
  std::vector<cl::Device> get_devices(const std::string &vendor_name) {
      size_t i;
      cl_int err;
      std::vector<cl::Platform> platforms;
      OCL_CHECK(err, err = cl::Platform::get(&platforms));
      cl::Platform platform;
      for (i = 0; i < platforms.size(); i++) {
          platform = platforms[i];
          OCL_CHECK(err,
                    std::string platformName =
                        platform.getInfo<CL_PLATFORM_NAME>(&err));
          if (platformName == vendor_name) {
              std::cout << "Found Platform" << std::endl;
              std::cout << "Platform Name: " << platformName.c_str() << std::endl;
              break;
          }
      }
      if (i == platforms.size()) {
          std::cout << "Error: Failed to find Xilinx platform" << std::endl;
          exit(EXIT_FAILURE);
      }
      //Getting ACCELERATOR Devices and selecting 1st such device
      std::vector<cl::Device> devices;
      OCL_CHECK(err,
                err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
      return devices;
  }

  std::vector<cl::Device> get_xil_devices() { return get_devices("Xilinx"); }
  
  std::vector<unsigned char> read_binary_file(const std::string &xclbin_file_name) {
      std::cout << "INFO: Reading " << xclbin_file_name << std::endl;

      if (access(xclbin_file_name.c_str(), R_OK) != 0) {
          printf("ERROR: %s xclbin not available please build\n",
                xclbin_file_name.c_str());
          exit(EXIT_FAILURE);
      }
      //Loading XCL Bin into char buffer
      std::cout << "Loading: '" << xclbin_file_name.c_str() << "'\n";
      std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
      bin_file.seekg(0, bin_file.end);
      auto nb = bin_file.tellg();
      bin_file.seekg(0, bin_file.beg);
      std::vector<unsigned char> buf;
      buf.resize(nb);
      bin_file.read(reinterpret_cast<char*>(buf.data()), nb);
      return buf;
  } 

  bool is_emulation() {
      bool ret = false;
      char *xcl_mode = getenv("XCL_EMULATION_MODE");
      if (xcl_mode != NULL) {
          ret = true;
      }
      return ret;
  }

  bool is_hw_emulation() {
      bool ret = false;
      char *xcl_mode = getenv("XCL_EMULATION_MODE");
      if ((xcl_mode != NULL) && !strcmp(xcl_mode, "hw_emu")) {
          ret = true;
      }
      return ret;
  }

  bool is_xpr_device(const char *device_name) {
      const char *output = strstr(device_name, "xpr");

      if (output == NULL) {
          return false;
      } else {
          return true;
      }
  }
    class Stream{
      public:
        static decltype(&clCreateStream) createStream;
        static decltype(&clReleaseStream) releaseStream;
        static decltype(&clReadStream) readStream;
        static decltype(&clWriteStream) writeStream;
        static decltype(&clPollStreams) pollStreams;
        static void init(const cl_platform_id& platform) {
            void *bar = clGetExtensionFunctionAddressForPlatform(platform, "clCreateStream");
            createStream = (decltype(&clCreateStream))bar;
            bar = clGetExtensionFunctionAddressForPlatform(platform, "clReleaseStream");
            releaseStream = (decltype(&clReleaseStream))bar;
            bar = clGetExtensionFunctionAddressForPlatform(platform, "clReadStream");
            readStream = (decltype(&clReadStream))bar;
            bar = clGetExtensionFunctionAddressForPlatform(platform, "clWriteStream");
            writeStream = (decltype(&clWriteStream))bar;
            bar = clGetExtensionFunctionAddressForPlatform(platform, "clPollStreams");
            pollStreams = (decltype(&clPollStreams))bar;
        }
    };
}