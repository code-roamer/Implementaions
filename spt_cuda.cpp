#include<torch/extension.h>
#include"spt_cuda_kernel.h"


//CUDA interfaces declarations

void bilinearSamplerBHWD_updateOutput_cuda(
	torch::Tensor& inputImages,
	torch::Tensor& grids,
	torch::Tensor& output);

void bilinearSamplerBHWD_updateGradInput_cuda(
	torch::Tensor& inputImages,
	torch::Tensor& grids,
	torch::Tensor& gradInputImages,
	torch::Tensor& gradGrids,
	torch::Tensor& gradOutput
);

void bilinearSamplerBHWD_updateGradInputOnlyGrid_cuda(
	torch::Tensor& inputImages,
	torch::Tensor& grids,
	torch::Tensor& gradGrids,
	torch::Tensor& gradOutput
);


//Check Tensor
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//Cplusplus interfaces

void bilinearSamplerBHWD_updateOutput(
	torch::Tensor& inputImages,
	torch::Tensor& grids,
	torch::Tensor& output)
{
	CHECK_INPUT(inputImages);
	CHECK_INPUT(grids);
	CHECK_INPUT(output);

	bilinearSamplerBHWD_updateOutput_cuda(inputImages, grids, output);
}

void bilinearSamplerBHWD_updateGradInput(
	torch::Tensor& inputImages,
	torch::Tensor& grids,
	torch::Tensor& gradInputImages,
	torch::Tensor& gradGrids,
	torch::Tensor& gradOutput)
{
	CHECK_INPUT(inputImages);
	CHECK_INPUT(grids);
	CHECK_INPUT(gradInputImages);
	CHECK_INPUT(gradGrids);
	CHECK_INPUT(gradOutput);

	bilinearSamplerBHWD_updateGradInput_cuda(inputImages, grids, gradInputImages, gradGrids, gradOutput);
}

void bilinearSamplerBHWD_updateGradInputOnlyGrid(
	torch::Tensor& inputImages,
	torch::Tensor& grids,
	torch::Tensor& gradGrids,
	torch::Tensor& gradOutput)
{
	CHECK_INPUT(inputImages);
	CHECK_INPUT(grids);
	CHECK_INPUT(gradGrids);
	CHECK_INPUT(gradOutput);

	bilinearSamplerBHWD_updateGradInputOnlyGrid_cuda(inputImages, grids, gradGrids, gradOutput);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &bilinearSamplerBHWD_updateOutput, "Spatial Transform forward");
	m.def("backward", &bilinearSamplerBHWD_updateGradInput, "Spatial Transform backward");
	m.def("backwardGrid", &bilinearSamplerBHWD_updateGradInputOnlyGrid, "Spatial Transform backward only Grids");
}
