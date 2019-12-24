#pragma once

#ifndef TVL1_CUDA
#define TVL1_CUDA

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;
class TVL1_cuda
{
public:
	TVL1_cuda();
	~TVL1_cuda(); 
	void initGpuMat(Size matSize);
	void process(Mat &frame0, Mat &frame1, Mat &flow);
private:
	Mat * gs_frame0;
	Mat * gs_frame1;
	GpuMat * d_frame0;
	GpuMat * d_frame1;
	GpuMat * d_flow;

	Ptr<cuda::OpticalFlowDual_TVL1> tvl1;

	bool isFirstTime;
};

#endif // !1
