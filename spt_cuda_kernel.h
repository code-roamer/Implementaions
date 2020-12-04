#pragma once
#include<torch/extension.h>

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