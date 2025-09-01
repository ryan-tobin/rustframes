#[cfg(feature = "cuda")]
pub mod cuda {
    use super::super::Array;
    use cuda::runtime::CudaDevice;
    use cudarc::driver::*;
    use cudarc::nvrtc::compile_ptx;
    use parquet::data_type::DataType;
    use rand::rand_core::block;
    use std::collections::HashMap;

    pub struct GpuArray {
        pub data: CudaSlice<f32>,
        pub shape: Vec<usize>,
        pub device: Arc<CudaDevice>,
    }

    impl GpuArray {
        /// Create GPU array from CPU array
        pub fn from_cpu_array(cpu_array: &Array<f64>) -> Result<Self, Box<dyn std::error::Error>> {
            let device = CudaDevice::new(0)?;

            let f32_data: Vec<f32> = cpu_array.data.iter().map(|&x| x as f32).collect();

            let gpu_data = device.htod_sync_copy(&f32_data);

            OK(GpuArray {
                data: gpu_data,
                shape: cpu_array.shape.clone(),
                device: Arc::new(device),
            })
        }

        /// Transfer back to CPU
        pub fn to_cpu_array(&self) -> Result<Array<f64>, Box<dyn std::error::Error>> {
            let cpu_data: Vec<f32> = self.device.dtoh_sync_copy(&self.data)?;
            let f64_data: Vec<f64> = cpu_data.iter().map(|&x| x as f64).collect();

            Ok(Array::from_vec(f64_data, self.shape.clone()))
        }

        /// GPU matrix multiplication using cuBLAS
        pub fn matmul_gpu(&self, other: &GpuArray) -> Result<GpuArray, Box<dyn std::error::Error>> {
            use cudarc::cublas::{CudaBlas, GemmConfig};

            assert_eq!(self.shape.len(), 2, "Left matrix must be 2D");
            assert_eq!(other.shape.len(), 2, "Right matrix must be 2D");
            assert_eq!(
                self.shape[1], other.shape[0],
                "Matrix dimensions incompatible"
            );

            let (m, k) = (self.shape[0], self.shape[1]);
            let n = other.shape[1];

            let blas = CudaBlas::new(self.device.clone())?;
            let mut result = self.device.alloc_zeros::<f32>(m * n)?;

            let config = GemmConfig {
                transa: cudarc::cublas::Transpose::NoTrans,
                transb: cudarc::cublas::Transpose::NoTrans,
                m: m as i32,
                n: n as i32,
                k: k as i32,
                alpha: 1.0,
                lda: k as i32,
                ldb: n as i32,
                beta: 0.0,
                ldc: n as i32,
            };

            unsafe {
                blas.gemm(config, &self.data, &other.data, &mut result)?;
            }

            Ok(GpuArray {
                data: result,
                shape: vec![m, n],
                device: self.device.clone(),
            })
        }

        /// Element-wise operation using custom CUDA kernels
        pub fn add_gpu(&self, other: &GpuArray) -> Result<GpuArray, Box<dyn std::error::Error>> {
            assert_eq!(self.shape, other.shape, "Arrays must have same shape");

            let kernel_src = r#"
            extern "C" __global__ void add_kernel(float* a, float* b, float* c, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    c[idx] = a[idx] + b[idx];
                }
            }
            "#;

            let ptx = compile_ptx(kernel_src)?;
            self.device.load_ptx(ptx, "add_kernel", &["add_kernel"])?;

            let kernel = self.device.get_func("add_kernel", "add_kernel")?;
            let mut result = self.device.alloc_zeros::<f32>(self.data.len())?;

            let n = self.data.len();
            let block_size = 256;
            let grid_size = (n * block_size - 1) / block_size;

            unsafe {
                kernel.launch(
                    LaunchConfig {
                        grid_dim: (grid_size as u32, 1, 1),
                        block_dim: (block_size as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&self.data, &other.data, &mut result, n as i32),
                )?;
            }

            self.device.synchronize()?;

            Ok(GpuArray {
                data: result,
                shape: self.shape.clone(),
                device: self.device.clone(),
            })
        }

        /// CUDA-accelerated reduction operations
        pub fn sum_gpu(&self) -> Result<f32, Box<dyn std::error::Error>> {
            let kernel_src = r#"
            extern "C" __global__ void sum_reduce_kernel(float* input, float* output, int n) {
                extern __shared__ float sdata[];

                int tid = threadIdx.x;
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                // Load data into shared memory
                sdata[tid] = (idx < n) ? input[idx] : 0.0f;
                __syncthreads();

                // Reduction in shared memory
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        sdata[tid] += sdata[tid + s];
                    }
                    __syncthreads();
                }

                // Write result of this block to global memory
                if (tid == 0) {
                    output[blockIdx.x] = sdata[0];
                }
            }
            "#;

            let ptx = compile_ptx(kernel_src)?;
            self.device
                .load_ptx(ptx, "sum_reduce", &["sum_reduce_kernel"])?;

            let kernel = self.device.get_func("sum_reduce", "sum_reduce_kernel")?;

            let n = self.data.len();
            let block_size = 256;
            let grid_size = (n * block_size - 1) / block_size;

            let mut partial_sums = self.device.alloc_zeros::<f32>(grid_size)?;

            unsafe {
                kernel.launch(
                    LaunchConfig {
                        grid_dim: (grid_size as u32, 1, 1),
                        block_dim: (block_size as u32, 1, 1),
                        shared_mem_bytes: block_size * std::mem::size_of::<f32>() as u32,
                    },
                    (&self.data, &mut partial_sums, n as i32),
                )?;
            }

            self.device.synchronize()?;

            let cpu_partial: Vec<f32> = self.device.dtoh_sync_copy(&partial_sums)?;
            Ok(cpu_partial.iter().sum())
        }

        /// GPU-accelerated convolution
        pub fn conv2d_gpu(
            &self,
            kernel: &GpuArray,
        ) -> Result<GpuArray, Box<dyn std::error::Error>> {
            use cudarc::cudnn::*;

            let cudnn = CudnnHandle::new(self.device.clone())?;

            let input_desc = TensorDescriptor::new(
                DataType::Float,
                TensorFormat::NCHW,
                &[1, 1, self.shape[0] as i32, self.shape[1] as i32],
            )?;

            let kernel_desc = FilterDescriptor::new(
                DataType::Float,
                TensorFormat::NCHW,
                &[1, 1, kernel.shape[0] as i32, kernel.shape[1] as i32],
            )?;

            let conv_desc = ConvolutionDescriptor::new(
                &[0, 0],
                &[1, 1],
                &[1, 1],
                ConvolutionMode::CrossCorrelation,
                DataType::Float,
            )?;

            let mut output_dims = [0i32; 4];
            cudnn.get_convolution_nd_forward_output_dim(
                &conv_desc,
                &input_desc,
                &kernel_desc,
                &mut output_dims,
            )?;

            let output_desc =
                TensorDescriptor::new(DataType::Float, TensorFormat::NCHW, &output_dims)?;

            let output_size = output_dims.iter().product::<i32>() as usize;
            let mut output = self.device.alloc_zeros::<f32>(output_size)?;

            let algo = cudnn.find_convolution_forward_algorithm(
                &input_desc,
                &self.data,
                &kernel_desc,
                &kernel.data,
                &conv_desc,
                &output_desc,
                &mut output,
            )?;

            cudnn.convolution_forward(
                1.0,
                &input_desc,
                &self.data,
                &kernel_desc,
                &kernel.data,
                &conv_desc,
                algo,
                0.0,
                &output_desc,
                &mut output,
            )?;

            Ok(GpuArray {
                data: output,
                shape: vec![output_dims[2] as usize, output_dims[3] as usize],
                device: self.device.clone(),
            })
        }
    }

    /// GPU-accelerated arrray operations
    impl Array<f64> {
        /// Send array to GPU for computation
        pub fn to_gpu(&self) -> Result<GpuArray, Box<dyn std::error::Error>> {
            GpuArray::from_cpu_array(self)
        }

        /// Perform GPU-accelerated matrix multiplication
        pub fn matmul_gpu(
            &self,
            other: &Array<f64>,
        ) -> Result<Array<f64>, Box<dyn std::error::Error>> {
            let gpu_a = self.to_gpu()?;
            let gpu_b = other.to_gpu()?;
            let gpu_result = gpu_a.matmul_gpu(&gpu_b)?;
            gpu_result.to_cpu_array()
        }

        /// GPU-accelerated element-wise addition
        pub fn add_gpu(
            &self,
            other: &Array<f64>,
        ) -> Result<Array<f64>, Box<dyn std::error::Error>> {
            let gpu_a = self.to_gpu()?;
            let gpu_b = other.to_gpu()?;
            let gpu_result = gpu_a.add_gpu(&gpu_b)?;
            gpu_result.to_cpu_array()
        }

        /// GPU-accelerated sum reduction
        pub fn sum_gpu(&self) -> Result<f64, Box<dyn std::error::Error>> {
            let gpu_array = self.to_gpu()?;
            let result = gpu_array.sum_gpu()?;
            Ok(result as f64)
        }
    }

    // Memory pool for efficient GPU memory management
    pub struct GpuMemoryPool {
        device: Arc<CudaDevice>,
        free_buffers: HashMap<usize, Vec<CudaSlice<f32>>>,
        used_buffers: Vec<CudaSlice<f32>>,
    }

    impl GpuMemoryPool {
        pub fn new(device: Arc<CudaDevice>) -> Self {
            GpuMemoryPool {
                device,
                free_buffers: HashMap::new(),
                used_buffers: Vec::new(),
            }
        }

        pub fn allocate(
            &mut self,
            size: usize,
        ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            if let Some(buffers) = self.free_buffers.get_mut(&size) {
                if let Some(buffer) = buffers.pop() {
                    self.used_buffers.push(buffer.clone());
                    return Ok(buffer);
                }
            }

            // Allocate new buffer
            let buffer = self.device.alloc_zeros::<f32>(size)?;
            self.used_buffers.push(buffer.clone());
            Ok(buffer)
        }

        pub fn deallocate(&mut self, buffer: CudaSlice<f32>) {
            let size = buffer.len();
            if let Some(pos) = self
                .used_buffers
                .iter()
                .position(|b| std::ptr::eq(b.as_ptr(), buffer.as_ptr()))
            {
                self.used_buffers.remove(pos);
                self.free_buffers
                    .entry(size)
                    .or_insert_with(Vec::new)
                    .push(buffer);
            }
        }

        pub fn clear(&mut self) {
            self.free_buffers.clear();
            self.used_buffers.clear();
        }
    }
}

// ROCm/HIP support for AMD GPUs
#[cfg(feature = "rocm")]
pub mod rocm {
    use super::super::Array;
    use hip_rs::*;

    pub struct HipArray {
        pub data: DeviceBuffer<f32>,
        pub shape: Vec<usize>,
        pub device: HipDevice,
    }

    impl HipArray {
        pub fn from_cpu_array(cpu_array: &Array<f64>) -> Result<Self, Box<dyn std::error::Error>> {
            let device = HipDevice::new(0)?;
            device.set_current()?;

            let f32_data: Vec<f32> = cpu_array.data.iter().map(|&x| x as f32).collect();
            let gpu_data = device.alloc_and_copy(&f32_data)?;

            Ok(HipArray {
                data: gpu_data,
                shape: cpu_array.shape.clone(),
                device,
            })
        }

        pub fn to_cpu_array(&self) -> Result<Array<f64>, Box<dyn std::error::Error>> {
            self.device.set_current()?;
            let cpu_data: Vec<f32> = self.device.copy_to_host(&self.data)?;
            let f64_data: Vec<f64> = cpu_data.iter().map(|&x| x as f64).collect();

            Ok(Array::from_vec(f64_data, self.shape.clone()))
        }

        pub fn matmul_rocm(
            &self,
            other: &HipArray,
        ) -> Result<HipArray, Box<dyn std::error::Error>> {
            use rocblas_rs::*;

            assert_eq!(self.shape.len(), 2, "Left matrix must be 2D");
            assert_eq!(other.shape.len(), 2, "Right matrix must be 2D");
            assert_eq!(
                self.shape[1], other.shape[0],
                "Matrix dimensions incompatible"
            );

            let (m, k) = (self.shape[0], self.shape[1]);
            let n = other.shape[1];

            self.device.set_current()?;
            let rocblas_handle = RocblasHandle::new()?;
            let mut result = self.device.alloc_zeros::<f32>(m * n)?;

            rocblas_handle.sgemm(
                RocblasOperation::None,
                RocblasOperation::None,
                m as i32,
                n as i32,
                k as i32,
                1.0, // alpha
                &self.data,
                k as i32, // lda
                &other.data,
                n as i32, // ldb
                0.0,      // beta
                &mut result,
                n as i32, // ldc
            )?;

            Ok(HipArray {
                data: result,
                shape: vec![m, n],
                device: self.device.clone(),
            })
        }
    }
}

// Metal Performance Shaders for Apple Silicon
#[cfg(feature = "metal")]
pub mod metal {
    use super::super::Array;
    use metal::*;
    use objc::rc::autoreleasepool;

    pub struct MetalArray {
        pub buffer: Buffer,
        pub shape: Vec<usize>,
        pub device: Device,
    }

    impl MetalArray {
        pub fn from_cpu_array(cpu_array: &Array<f64>) -> Result<Self, Box<dyn std::error::Error>> {
            let device = Device::system_default().ok_or("No Metal device found")?;

            let f32_data: Vec<f32> = cpu_array.data.iter().map(|&x| x as f32).collect();
            let buffer = device.new_buffer_with_data(
                f32_data.as_ptr() as *const std::ffi::c_void,
                (f32_data.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            Ok(MetalArray {
                buffer,
                shape: cpu_array.shape.clone(),
                device,
            })
        }

        pub fn to_cpu_array(&self) -> Result<Array<f64>, Box<dyn std::error::Error>> {
            let ptr = self.buffer.contents() as *const f32;
            let len = self.buffer.length() as usize / std::mem::size_of::<f32>();

            let f32_data = unsafe { std::slice::from_raw_parts(ptr, len) };
            let f64_data: Vec<f64> = f32_data.iter().map(|&x| x as f64).collect();

            Ok(Array::from_vec(f64_data, self.shape.clone()))
        }

        pub fn matmul_metal(
            &self,
            other: &MetalArray,
        ) -> Result<MetalArray, Box<dyn std::error::Error>> {
            use metal_performance_shaders::*;

            assert_eq!(self.shape.len(), 2, "Left matrix must be 2D");
            assert_eq!(other.shape.len(), 2, "Right matrix must be 2D");
            assert_eq!(
                self.shape[1], other.shape[0],
                "Matrix dimensions incompatible"
            );

            let (m, k) = (self.shape[0], self.shape[1]);
            let n = other.shape[1];

            autoreleasepool(|| {
                let command_queue = self.device.new_command_queue();
                let command_buffer = command_queue.new_command_buffer();

                // Create matrix descriptors
                let desc_a = MPSMatrixDescriptor::matrix_descriptor(
                    m as NSUInteger,
                    k as NSUInteger,
                    k as NSUInteger,
                    MPSDataType::Float32,
                );

                let desc_b = MPSMatrixDescriptor::matrix_descriptor(
                    k as NSUInteger,
                    n as NSUInteger,
                    n as NSUInteger,
                    MPSDataType::Float32,
                );

                let desc_c = MPSMatrixDescriptor::matrix_descriptor(
                    m as NSUInteger,
                    n as NSUInteger,
                    n as NSUInteger,
                    MPSDataType::Float32,
                );

                // Create result buffer
                let result_buffer = self.device.new_buffer(
                    (m * n * std::mem::size_of::<f32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                // Create MPS matrices
                let matrix_a = MPSMatrix::new(&self.buffer, &desc_a);
                let matrix_b = MPSMatrix::new(&other.buffer, &desc_b);
                let matrix_c = MPSMatrix::new(&result_buffer, &desc_c);

                // Perform matrix multiplication
                let matmul =
                    MPSMatrixMultiplication::new(&self.device, false, false, m, n, k, 1.0, 0.0);
                matmul.encode_to_command_buffer(&command_buffer, &matrix_a, &matrix_b, &matrix_c);

                command_buffer.commit();
                command_buffer.wait_until_completed();

                Ok(MetalArray {
                    buffer: result_buffer,
                    shape: vec![m, n],
                    device: self.device.clone(),
                })
            })
        }
    }
}

// Unified GPU interface
pub enum GpuBackend {
    #[cfg(feature = "cuda")]
    Cuda(cuda::GpuArray),
    #[cfg(feature = "rocm")]
    Rocm(rocm::HipArray),
    #[cfg(feature = "metal")]
    Metal(metal::MetalArray),
    Cpu, // Fallback to CPU
}

impl GpuBackend {
    pub fn detect_best_backend() -> Self {
        #[cfg(feature = "cuda")]
        {
            if cudarc::driver::CudaDevice::new(0).is_ok() {
                return GpuBackend::Cpu; // Placeholder - would create CUDA backend
            }
        }

        #[cfg(feature = "rocm")]
        {
            if hip_rs::HipDevice::new(0).is_ok() {
                return GpuBackend::Cpu; // Placeholder - would create ROCm backend
            }
        }

        #[cfg(feature = "metal")]
        {
            if metal::Device::system_default().is_some() {
                return GpuBackend::Cpu; // Placeholder - would create Metal backend
            }
        }

        GpuBackend::Cpu
    }

    pub fn matmul(&self, other: &GpuBackend) -> Result<GpuBackend, Box<dyn std::error::Error>> {
        match (self, other) {
            #[cfg(feature = "cuda")]
            (GpuBackend::Cuda(a), GpuBackend::Cuda(b)) => Ok(GpuBackend::Cuda(a.matmul_gpu(b)?)),
            #[cfg(feature = "rocm")]
            (GpuBackend::Rocm(a), GpuBackend::Rocm(b)) => Ok(GpuBackend::Rocm(a.matmul_rocm(b)?)),
            #[cfg(feature = "metal")]
            (GpuBackend::Metal(a), GpuBackend::Metal(b)) => {
                Ok(GpuBackend::Metal(a.matmul_metal(b)?))
            }
            _ => Err("Incompatible GPU backends or CPU fallback needed".into()),
        }
    }
}

pub struct GpuTuner {
    device_info: DeviceInfo,
    optimal_configs: std::collections::HashMap<String, TuningConfig>,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub memory_gb: f32,
    pub compute_capability: String,
    pub max_threads_per_block: u32,
    pub max_shared_memory: u32,
}

#[derive(Debug, Clone)]
pub struct TuningConfig {
    pub block_size: u32,
    pub grid_size_multiplier: f32,
    pub shared_memory_usage: f32,
    pub register_usage: f32,
}

impl Default for GpuTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuTuner {
    pub fn new() -> Self {
        GpuTuner {
            device_info: Self::query_device_info(),
            optimal_configs: std::collections::HashMap::new(),
        }
    }

    fn query_device_info() -> DeviceInfo {
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = cudarc::driver::CudaDevice::new(0) {
                return DeviceInfo {
                    name: device.name().unwrap_or_default(),
                    memory_gb: device.total_memory().unwrap_or(0) as f32 / 1e9,
                    compute_capability: format!(
                        "{}.{}",
                        device.compute_capability().0,
                        device.compute_capability().1
                    ),
                    max_threads_per_block: 1024, // Common for modern GPUs
                    max_shared_memory: 48 * 1024, // 48KB for modern GPUs
                };
            }
        }

        // Fallback device info
        DeviceInfo {
            name: "Unknown".to_string(),
            memory_gb: 0.0,
            compute_capability: "0.0".to_string(),
            max_threads_per_block: 256,
            max_shared_memory: 16 * 1024,
        }
    }

    pub fn tune_matmul(&mut self, m: usize, n: usize, k: usize) -> TuningConfig {
        let key = format!("matmul_{}x{}x{}", m, n, k);

        if let Some(config) = self.optimal_configs.get(&key) {
            return config.clone();
        }

        // Auto-tune based on matrix size and device capabilities
        let block_size = if m * n > 1_000_000 {
            self.device_info.max_threads_per_block
        } else if m * n > 100_000 {
            512
        } else {
            256
        };

        let config = TuningConfig {
            block_size,
            grid_size_multiplier: 1.0,
            shared_memory_usage: 0.8, // Use 80% of available shared memory
            register_usage: 0.7,      // Conservative register usage
        };

        self.optimal_configs.insert(key, config.clone());
        config
    }

    pub fn benchmark_configuration(
        &self,
        _config: &TuningConfig,
        _operation: &str,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // Placeholder for actual benchmarking
        Ok(1.0) // milliseconds
    }
}
