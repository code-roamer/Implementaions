#include<torch/all.h>

#include<cuda.h>
#include<cuda_runtime.h>

//CUDA kernel code

__device__ bool between(int x, int lbound, int ubound)
{
	return x >= lbound && x <= ubound;
}

template<typename scalar_t>
__global__ void bilinearSamplerBHWD_updateOutput_kernel(
	scalar_t* inputImages, int inputImages_strideBatch, int inputImages_strideHeight, int inputImages_strideWidth,
	scalar_t* grids, int grids_strideBatch, int grids_strideHeight, int grids_strideWidth,
	scalar_t* output, int output_strideBatch, int output_strideHeight, int output_strideWidth,
	int inputImages_height, int inputImages_width,
	int output_channel, int output_height, int output_width)
{
	const int xOut = blockIdx.x * blockDim.y + threadIdx.y;
	const int yOut = blockIdx.y;
	const int b = blockIdx.z;
	
	const bool withInOutput = xOut < output_width;
	const bool withInGrid = blockIdx.x * blockDim.y + threadIdx.x / 2 < output_width;

	__shared__ scalar_t gridData[32];
	if (threadIdx.y == 0 && withInGrid)
	{
		gridData[threadIdx.x] = grids[b * grids_strideBatch + yOut * grids_strideHeight +
			xOut * grids_strideWidth + threadIdx.x];
	}

	__syncthreads();

	if (!withInOutput) return;

	//-1<=xs<=1, -1<=ys<=1 must be satisfied
	scalar_t xs = (gridData[threadIdx.y * 2] + 1) * (inputImages_width - 1) / 2;
	scalar_t ys = (gridData[threadIdx.y * 2 + 1] + 1) * (inputImages_height - 1) / 2;
	int xTopLeft = int(floor(xs)), yTopLeft = int(floor(ys));
	scalar_t xTopLeftWeight = 1 - (xs - xTopLeft), yTopLeftWeight = 1 - (ys - yTopLeft);

	bool topLeftWithInImage = between(xTopLeft, 0, inputImages_width - 1) &&
		between(yTopLeft, 0, inputImages_height - 1);
	bool topRightWithInImage = between(xTopLeft + 1, 0, inputImages_width - 1) &&
		between(yTopLeft, 0, inputImages_height - 1);
	bool bottomLeftWithInImage = between(xTopLeft, 0, inputImages_width - 1) &&
		between(yTopLeft + 1, 0, inputImages_height - 1);
	bool bottomRightWithInImage = between(xTopLeft + 1, 0, inputImages_width - 1) &&
		between(yTopLeft + 1, 0, inputImages_height - 1);


	int inputTopLeft_Address = b * inputImages_strideBatch + 
		yTopLeft * inputImages_strideHeight + xTopLeft * inputImages_strideWidth;
	int inputTopRight_Address = inputTopLeft_Address + inputImages_strideWidth;
	int inputBottomLeft_Address = inputTopLeft_Address + inputImages_strideHeight;
	int inputBottomRight_Address = inputBottomLeft_Address + inputImages_strideWidth;
	int output_Address = b * output_strideBatch + yOut * output_strideHeight + xOut * output_strideWidth;

	scalar_t inTopLeft = 0, inTopRight = 0, inBottomLeft = 0, inBottomRight = 0;

	for (int t = threadIdx.x; t < output_channel; t += blockDim.x)
	{
		if (topLeftWithInImage) inTopLeft = inputImages[inputTopLeft_Address + t];
		if(topRightWithInImage) inTopRight = inputImages[inputTopRight_Address + t];
		if (bottomLeftWithInImage) inBottomLeft = inputImages[inputBottomLeft_Address + t];
		if (bottomRightWithInImage) inBottomRight = inputImages[inputBottomRight_Address + t];
		output[output_Address + t] = inTopLeft * xTopLeftWeight * yTopLeftWeight +
			inTopRight * (1 - xTopLeftWeight) * yTopLeftWeight +
			inBottomLeft * xTopLeftWeight * (1 - yTopLeftWeight) +
			inBottomRight * (1 - xTopLeftWeight) * (1 - yTopLeftWeight);
	}
}


template<typename scalar_t>
__global__ void bilinearSamplerBHWD_updateGradInputOnlyGrid_kernel(
	scalar_t* inputImages, int inputImages_strideBatch, int inputImages_strideHeight, int inputImages_strideWidth,
	scalar_t* grids, int grids_strideBatch, int grids_strideHeight, int grids_strideWidth,
	scalar_t* gradOutput, int output_strideBatch, int output_strideHeight, int output_strideWidth,
	scalar_t* gradGrids, int output_height, int output_width, int output_channel,
	int inputImages_height, int inputImages_width)
{
	const int xOut = blockIdx.x * blockDim.y + threadIdx.y;
	const int yOut = blockIdx.y;
	const int b = blockIdx.z;

	const bool withInOutput = xOut < output_width;
	const bool withInGrid = blockIdx.x * blockDim.y + threadIdx.x / 2 < output_width;

	__shared__ scalar_t gridData[32];
	if (threadIdx.y == 0 && withInGrid) {
		gridData[threadIdx.x] = grids[b * grids_strideBatch + yOut * grids_strideHeight +
			xOut * grids_strideWidth + threadIdx.x];
	}
	__syncthreads();

	scalar_t xf = (gridData[threadIdx.y * 2] + 1) * (inputImages_width - 1) / 2;
	scalar_t yf = (gridData[threadIdx.y * 2 + 1] + 1) * (inputImages_height - 1) / 2;
	int xTopLeft = int(floor(xf)), yTopLeft = int(floor(yf));
	scalar_t xTopLeftWeight = (1 - (xTopLeft - xf)), yTopLeftWeight = (1 - (yTopLeft - yf));

	bool topLeftWithInImage = between(xTopLeft, 0, inputImages_width - 1) &&
		between(yTopLeft, 0, inputImages_height - 1);
	bool topRightWithInImage = between(xTopLeft + 1, 0, inputImages_width - 1) &&
		between(yTopLeft, 0, inputImages_height - 1);
	bool bottomLeftWithInImage = between(xTopLeft, 0, inputImages_width - 1) &&
		between(yTopLeft + 1, 0, inputImages_height - 1);
	bool bottomRightWithInImage = between(xTopLeft + 1, 0, inputImages_width - 1) &&
		between(yTopLeft + 1, 0, inputImages_height - 1);

	int inTopLeft_Address = b * inputImages_strideBatch +
		xTopLeft * inputImages_strideWidth + yTopLeft * inputImages_strideHeight;
	int inTopRight_Address = inTopLeft_Address + inputImages_strideWidth;
	int inBottomLeft_Address = inTopLeft_Address + inputImages_strideHeight;
	int inBottomRight_Address = inBottomLeft_Address + inputImages_strideWidth;
	int grids_Address = b * grids_strideBatch + yOut * grids_strideHeight + xOut * grids_strideWidth;
	int output_Address = b * output_strideBatch + yOut * output_strideHeight + xOut * output_strideWidth;


	__syncthreads();
	if (threadIdx.x == 0) {
		gridData[threadIdx.y * 2] = 0;
		gridData[threadIdx.y * 2 + 1] = 0;
	}

	scalar_t gradxSum = 0.0, gradySum = 0.0;
	scalar_t inTopLeft = 0, inTopRight = 0, inBottomLeft = 0, inBottomRight = 0;

	if (withInOutput) {
		for (int t = threadIdx.x; t < output_channel; t += blockDim.x)
		{
			if (topLeftWithInImage) inTopLeft = inputImages[inTopLeft_Address + t];
			if (topRightWithInImage) inTopRight = inputImages[inTopRight_Address + t];
			if (bottomLeftWithInImage) inBottomLeft = inputImages[inBottomLeft_Address + t];
			if (bottomRightWithInImage) inBottomRight = inputImages[inBottomRight_Address + t];

			float vx = -inTopLeft * yTopLeftWeight +
				inTopRight * yTopLeftWeight -
				inBottomLeft * (1 - yTopLeftWeight) +
				inBottomRight * (1 - yTopLeftWeight);

			float vy = -inTopLeft * xTopLeftWeight -
				inTopRight * (1 - xTopLeftWeight) +
				inBottomLeft * xTopLeftWeight +
				inBottomRight * (1 - xTopLeftWeight);

			gradxSum += gradOutput[output_Address + t] * vx * (output_width - 1) / 2;
			gradySum += gradOutput[output_Address + t] * vy * (output_height - 1) / 2;
		}
	}

	__syncthreads();
	atomicAdd(&gridData[threadIdx.y * 2], gradxSum);
	atomicAdd(&gridData[threadIdx.y * 2 + 1], gradySum);

	__syncthreads();
	if (threadIdx.x == 0 && withInGrid) {
		gradGrids[grids_Address] = gridData[threadIdx.y * 2];
		gradGrids[grids_Address] = gridData[threadIdx.y * 2 + 1];
	}
}

template<typename scalar_t>
__global__ void bilinearSamplerBHWD_updateGradInput_kernel(
	scalar_t* inputImages, int inputImages_strideBatch, int inputImages_strideHeight, int inputImages_strideWidth,
	scalar_t* grids, int grids_strideBatch, int grids_strideHeight, int grids_strideWidth,
	scalar_t* gradOutput, int output_strideBatch, int output_strideHeight, int output_strideWidth,
	scalar_t* gradInputImages, scalar_t* gradGrids, int output_height, int output_width, int output_channel,
	int inputImages_height, int inputImages_width)
{
	const int xOut = blockIdx.x * blockDim.y + threadIdx.y;
	const int yOut = blockIdx.y;
	const int b = blockIdx.z;

	const bool withInOutput = xOut < output_width;
	const bool withInGrid = blockIdx.x * blockDim.y + threadIdx.x / 2 < output_width;

	__shared__ scalar_t gridData[32];
	if (threadIdx.y == 0 && withInGrid) {
		gridData[threadIdx.x] = grids[b * grids_strideBatch + yOut * grids_strideHeight +
			xOut * grids_strideWidth + threadIdx.x];
	}
	__syncthreads();

	scalar_t xf = (gridData[threadIdx.y * 2] + 1) * (inputImages_width - 1) / 2;
	scalar_t yf = (gridData[threadIdx.y * 2 + 1] + 1) * (inputImages_height - 1) / 2;
	int xTopLeft = int(floor(xf)), yTopLeft = int(floor(yf));
	scalar_t xTopLeftWeight = (1 - (xTopLeft - xf)), yTopLeftWeight = (1 - (yTopLeft - yf));

	bool topLeftWithInImage = between(xTopLeft, 0, inputImages_width - 1) &&
		between(yTopLeft, 0, inputImages_height - 1);
	bool topRightWithInImage = between(xTopLeft + 1, 0, inputImages_width - 1) &&
		between(yTopLeft, 0, inputImages_height - 1);
	bool bottomLeftWithInImage = between(xTopLeft, 0, inputImages_width - 1) &&
		between(yTopLeft + 1, 0, inputImages_height - 1);
	bool bottomRightWithInImage = between(xTopLeft + 1, 0, inputImages_width - 1) &&
		between(yTopLeft + 1, 0, inputImages_height - 1);

	int inTopLeft_Address = b * inputImages_strideBatch +
		xTopLeft * inputImages_strideWidth + yTopLeft * inputImages_strideHeight;
	int inTopRight_Address = inTopLeft_Address + inputImages_strideWidth;
	int inBottomLeft_Address = inTopLeft_Address + inputImages_strideHeight;
	int inBottomRight_Address = inBottomLeft_Address + inputImages_strideWidth;
	int grids_Address = b * grids_strideBatch + yOut * grids_strideHeight + xOut * grids_strideWidth;
	int output_Address = b * output_strideBatch + yOut * output_strideHeight + xOut * output_strideWidth;

	__syncthreads();
	if (threadIdx.x == 0) {
		gridData[threadIdx.y * 2] = 0;
		gridData[threadIdx.y * 2 + 1] = 0;
	}

	scalar_t gradxSum = 0.0, gradySum = 0.0;
	scalar_t inTopLeft = 0, inTopRight = 0, inBottomLeft = 0, inBottomRight = 0;

	if (withInOutput) {
		for (int t = threadIdx.x; t < output_channel; t += blockDim.x)
		{
			if (topLeftWithInImage)
			{
				inTopLeft = inputImages[inTopLeft_Address + t];
				atomicAdd(&gradInputImages[inTopLeft_Address + t], gradOutput[output_Address] * xTopLeftWeight * yTopLeftWeight);
			}
			if (topRightWithInImage)
			{
				inTopRight = inputImages[inTopRight_Address + t];
				atomicAdd(&gradInputImages[inTopRight_Address + t], gradOutput[output_Address] * (1 - xTopLeftWeight) * yTopLeftWeight);
			}
			if (bottomLeftWithInImage)
			{
				inBottomLeft = inputImages[inBottomLeft_Address + t];
				atomicAdd(&gradInputImages[inBottomLeft_Address + t], gradOutput[output_Address] * xTopLeftWeight * (1 - yTopLeftWeight));
			}
			if (bottomRightWithInImage)
			{
				inBottomRight = inputImages[inBottomRight_Address + t];
				atomicAdd(&gradInputImages[inBottomRight_Address + t], gradOutput[output_Address] * (1 - xTopLeftWeight) * (1 - yTopLeftWeight));
			}

			float vx = -inTopLeft * yTopLeftWeight +
				inTopRight * yTopLeftWeight -
				inBottomLeft * (1 - yTopLeftWeight) +
				inBottomRight * (1 - yTopLeftWeight);

			float vy = -inTopLeft * xTopLeftWeight -
				inTopRight * (1 - xTopLeftWeight) +
				inBottomLeft * xTopLeftWeight +
				inBottomRight * (1 - xTopLeftWeight);

			gradxSum += gradOutput[output_Address + t] * vx * (output_width - 1) / 2;
			gradySum += gradOutput[output_Address + t] * vy * (output_height - 1) / 2;
		}
	}

	__syncthreads();
	atomicAdd(&gridData[threadIdx.y * 2], gradxSum);
	atomicAdd(&gridData[threadIdx.y * 2 + 1], gradySum);

	__syncthreads();
	if (threadIdx.x == 0 && withInGrid) {
		gradGrids[grids_Address] = gridData[threadIdx.y * 2];
		gradGrids[grids_Address] = gridData[threadIdx.y * 2 + 1];
	}
}

void bilinearSamplerBHWD_updateOutput_cuda(
	torch::Tensor& inputImages,
	torch::Tensor& grids,
	torch::Tensor& output)
{
	//(ceil(W/16), H, B), (32, 16), 32 for store grid coordination
	dim3 blocks((output.size(2) + 15) / 16, output.size(1), output.size(0));
	dim3 threads(32, 16);
	int inputImages_strideBatch = inputImages.stride(0), inputImages_strideHeight = inputImages.stride(1), inputImages_strideWidth = inputImages.stride(2);
	int grids_strideBatch = grids.stride(0), grids_strideHeight = grids.stride(1), grids_strideWidth = grids.stride(2);
	int output_strideBatch = output.stride(0), output_strideHeight = output.stride(1), output_strideWidth = output.stride(2);
	int inputImages_height = inputImages.size(1), inputImages_width = inputImages.size(2), inputImages_channel = inputImages.size(3);
	int output_height = output.size(1), output_width = output.size(2);
	/*bilinearSamplingFromGrid<<<blocks, threads>>> (
		inputImages.data<float>(), inputImages_strideBatch, inputImages_strideChannel, inputImages_strideHeight, inputImages_strideWidth,
		grids.data<float>(), grids_strideBatch, grids_strideXY, grids_strideHeight, grids_strideWidth,
		output.data<float>(), output_strideBatch, output_strideChannel, output_strideHeight, output_strideWidth,
		inputImages_Height, inputImages_Width, inputImages_Channel, inputImages_Width);*/
	AT_DISPATCH_FLOATING_TYPES(grids.type(), "bilinearSamplerBHWD_updateOutput_cuda", ([&] {
		bilinearSamplerBHWD_updateOutput_kernel<scalar_t><<<blocks, threads>>> (
			inputImages.data<scalar_t>(), inputImages_strideBatch, inputImages_strideHeight, inputImages_strideWidth,
			grids.data<scalar_t>(), grids_strideBatch, grids_strideHeight, grids_strideWidth,
			output.data<scalar_t>(), output_strideBatch, output_strideHeight, output_strideWidth,
			inputImages_height, inputImages_width, inputImages_channel, output_height, output_width);
	}));
}


void bilinearSamplerBHWD_updateGradInput_cuda(
	torch::Tensor& inputImages,
	torch::Tensor& grids,
	torch::Tensor& gradInputImages,
	torch::Tensor& gradGrids,
	torch::Tensor& gradOutput)
{
	dim3 blocks((gradOutput.size(2) + 15) / 16, gradOutput.size(1), gradOutput.size(0));
	dim3 threads(32, 16);

	int inputImages_strideBatch = inputImages.stride(0), inputImages_strideChannel = inputImages.stride(3), inputImages_strideHeight = inputImages.stride(1), inputImages_strideWidth = inputImages.stride(2);
	int grids_strideBatch = grids.stride(0), grids_strideXY = grids.stride(3), grids_strideHeight = grids.stride(1), grids_strideWidth = grids.stride(2);
	int output_strideBatch = gradOutput.stride(0), output_strideChannel = gradOutput.stride(3), output_strideHeight = gradOutput.stride(1), output_strideWidth = gradOutput.stride(2);
	int output_height = gradOutput.size(1), output_width = gradOutput.size(2), output_channel = gradOutput.size(3);
	int inputImages_height = inputImages.size(1), inputImages_width = inputImages.size(2);

	AT_DISPATCH_FLOATING_TYPES(grids.type(), "bilinearSamplerBHWD_updateGradInput_cuda", ([&] {
		bilinearSamplerBHWD_updateGradInput_kernel<scalar_t> <<<blocks, threads>>> (
			inputImages.data<scalar_t>(),inputImages_strideBatch,inputImages_strideHeight,inputImages_strideWidth,
			grids.data<scalar_t>(), grids_strideBatch,grids_strideHeight,grids_strideWidth,
			gradOutput.data<scalar_t>(), output_strideBatch,output_strideHeight,output_strideWidth,
			gradInputImages.data<scalar_t>(), gradGrids.data<scalar_t>(), output_height,output_width,output_channel,
			inputImages_height, inputImages_width);
	}));
}


void bilinearSamplerBHWD_updateGradInputOnlyGrid_cuda(
	torch::Tensor& inputImages,
	torch::Tensor& grids,
	torch::Tensor& gradGrids,
	torch::Tensor& gradOutput)
{
	dim3 blocks((gradOutput.size(2) + 15) / 16, gradOutput.size(1), gradOutput.size(0));
	dim3 threads(32, 16);
	int inputImages_strideBatch = inputImages.stride(0), inputImages_strideChannel = inputImages.stride(3), inputImages_strideHeight = inputImages.stride(1), inputImages_strideWidth = inputImages.stride(2);
	int grids_strideBatch = grids.stride(0), grids_strideXY = grids.stride(3), grids_strideHeight = grids.stride(1), grids_strideWidth = grids.stride(2);
	int output_strideBatch = gradOutput.stride(0), output_strideChannel = gradOutput.stride(3), output_strideHeight = gradOutput.stride(1), output_strideWidth = gradOutput.stride(2);
	int output_height = gradOutput.size(1), output_width = gradOutput.size(2), output_channel = gradOutput.size(3);
	int inputImages_height = inputImages.size(1), inputImages_width = inputImages.size(2);

	AT_DISPATCH_FLOATING_TYPES(grids.type(), "bilinearSampler_updateOutputOnlyGrid_cuda", ([&] {
		bilinearSamplerBHWD_updateGradInputOnlyGrid_kernel<scalar_t><<<blocks, threads>>> (
			inputImages.data<scalar_t>(),inputImages_strideBatch,inputImages_strideHeight,inputImages_strideWidth,
			grids.data<scalar_t>(), grids_strideBatch,grids_strideHeight,grids_strideWidth,
			gradOutput.data<scalar_t>(), output_strideBatch,output_strideHeight,output_strideWidth,
			gradGrids.data<scalar_t>(), output_height,output_width,output_channel,
			inputImages_height, inputImages_width);
	}));
}