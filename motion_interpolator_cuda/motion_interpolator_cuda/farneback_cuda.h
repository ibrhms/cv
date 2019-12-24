#pragma once

#ifndef FARNEBACK_CUDA
#define FARNEBACK_CUDA

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

class farneback_cuda
{
public:
	farneback_cuda();
	~farneback_cuda();

	void initGpuMat(Size matSize);
	void process(Mat &frame0, Mat &frame1, Mat &flow);
private:
	Mat * gs_frame0;
	Mat * gs_frame1;
	GpuMat * d_frame0;
	GpuMat * d_frame1;
	GpuMat * d_flow;

	Ptr<cuda::FarnebackOpticalFlow> farn;

	bool isFirstTime;
};

#endif // !1
