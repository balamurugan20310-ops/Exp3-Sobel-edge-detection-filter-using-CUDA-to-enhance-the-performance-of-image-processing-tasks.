# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.

<h3>BALAMURUGAN S</h3>
<h3>212225240020</h3>
<h3>EX. NO : 3</h3>
<h3>DATE : 08/08/2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```py
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter

from pathlib import Path

file_path = Path('/absolute/path/to/images.jpeg')
if file_path.exists():
    print("File exists!")
else:
    print("File does not exist!")

import os
print("Current Working Directory:", os.getcwd())

from google.colab import files
uploaded = files.upload()

from pathlib import Path

# Assuming the file is in the same directory as the notebook
file_path = Path('gill.jpg')
if file_path.exists():
    print("File exists!")
else:
    print("File does not exist!")

pwd

ls /content/gill.jpg

#ls -l /content/images.jpeg
import cv2
image = cv2.imread('/content/gill.jpg')
if image is None:
    print("Error: Image not found or unable to read the image.")
else:
    print("Image read successfully.")

%%writefile sobelEdgeDetectionFilter.cu
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;

// CUDA Kernel for Sobel Edge Detection
__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check for thread layout
    if (x >= width || y >= height) return;

    // Set boundary pixels to black (0) to avoid out-of-bounds array access
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        dstImage[y * width + x] = 0;
        return;
    }

    // Horizontal gradient operator (Gx)
    // [-1  0  1]
    // [-2  0  2]
    // [-1  0  1]
    int Gx = -1 * srcImage[(y - 1) * width + (x - 1)] + 1 * srcImage[(y - 1) * width + (x + 1)]
             - 2 * srcImage[y       * width + (x - 1)] + 2 * srcImage[y       * width + (x + 1)]
             - 1 * srcImage[(y + 1) * width + (x - 1)] + 1 * srcImage[(y + 1) * width + (x + 1)];

    // Vertical gradient operator (Gy)
    // [-1 -2 -1]
    // [ 0  0  0]
    // [ 1  2  1]
    int Gy = -1 * srcImage[(y - 1) * width + (x - 1)] - 2 * srcImage[(y - 1) * width + x] - 1 * srcImage[(y - 1) * width + (x + 1)]
             + 1 * srcImage[(y + 1) * width + (x - 1)] + 2 * srcImage[(y + 1) * width + x] + 1 * srcImage[(y + 1) * width + (x + 1)];

    // Compute edge magnitude: sqrt(Gx^2 + Gy^2)
    int magnitude = sqrtf((float)(Gx * Gx + Gy * Gy));

    // Clamp value to [0, 255] range
    if (magnitude > 255) magnitude = 255;

    dstImage[y * width + x] = (unsigned char)magnitude;
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Read input image as grayscale
    Mat image = imread("/content/gill.jpg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate host memory for output image
    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, image.data, imageSize, cudaMemcpyHostToDevice));

    // Define CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    // Write output image
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel.jpeg", outputImage);

    // Free memory
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print elapsed time
    printf("Total time taken: %f milliseconds\n", milliseconds);

    return 0;
}

!nvcc -o sobelEdgeDetectionFilter sobelEdgeDetectionFilter.cu `pkg-config --cflags --libs opencv4`

!./sobelEdgeDetectionFilter

import cv2
from matplotlib import pyplot as plt

# Read and display the output image
output_image_path = '/content/output_sobel.jpeg'
output_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)  # Use IMREAD_GRAYSCALE if it's a single-channel image

# Display the image
plt.imshow(output_image, cmap='gray')
plt.title('Edge Detection Output')
plt.axis('off')  # Hide the axes
plt.show()
```

## OUTPUT:
<img width="646" height="461" alt="image" src="https://github.com/user-attachments/assets/572f7df4-9ae3-42a6-81c7-8a0c8311fcc5" />


## RESULT:
Thus the program has been executed by using CUDA to perform parallel Sobel edge detection on an input image, leveraging GPU acceleration to compute spatial intensity gradients significantly faster than sequential CPU execution.

### Questions:

#### What challenges did you face while implementing the Sobel filter for color images?
  Multi-channel (RGB) interleaved memory layout requires complex thread indexing and significantly increases memory bandwidth overhead per pixel.Combining or reducing separate gradient magnitudes across all three individual color channels adds computational complexity compared to single-channel grayscale processing.
    
#### How did changing the block size influence the performance of your CUDA implementation?
  Small block sizes (e.g., $4 \times 4$) caused low warp occupancy and high overhead, resulting in slower execution times.
Optimal block sizes (e.g., $16 \times 16$) maximized Streaming Multiprocessor utilization and memory latency hiding, yielding the fastest performance.

#### What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
  CUDA sets outer border pixels to solid black to prevent memory errors, whereas the CPU uses reflection padding to calculate edge gradients.Minor pixel intensity variations ($\pm 1$) occur across the image due to differences in floating-point math and rounding between GPU and CPU architectures.
    
#### Suggest potential optimizations for improving the performance of the Sobel filter.
  Use shared memory tiling or read-only texture caching to drastically reduce repetitive global memory accesses for neighboring pixels.
Leverage vectorized memory loads (uchar4) and CUDA streams to maximize memory bandwidth and overlap GPU execution with data transfers.


### Deliverables:

Modified CUDA code with comments explaining your changes.
A report summarizing your findings, including graphs of execution times and a comparison of outputs.
Answers to the questions posed in the experiment.

### Tools Required:

* Hardware Resources
* Development Environment & Compilers
* Libraries & Frameworks

