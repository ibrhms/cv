#pragma once

#ifndef BROX_CUDA
#define BROX_CUDA

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;
	
class brox_cuda
{
public:
	brox_cuda();
	~brox_cuda(); 
	void initGpuMat(Size matSize);
	void process(Mat &frame0, Mat &frame1, Mat &flow);
private:
	Mat * gs_frame0;
	Mat * gs_frame1;
	GpuMat * d_frame0;
	GpuMat * d_frame1;
	GpuMat * d_flow;
	GpuMat *  d_frame0f;
	GpuMat * d_frame1f;

	Ptr<cuda::BroxOpticalFlow> brox;

	bool isFirstTime;
};

#endif