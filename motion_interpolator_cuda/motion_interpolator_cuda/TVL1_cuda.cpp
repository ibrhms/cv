#include "TVL1_cuda.h"



TVL1_cuda::TVL1_cuda()
{
	tvl1 = cuda::OpticalFlowDual_TVL1::create();
	isFirstTime = true;
}

TVL1_cuda::~TVL1_cuda()
{
	if (d_flow) delete d_flow;
	if (d_frame0) delete d_frame0;
	if (d_frame1) delete d_frame1;
	if (gs_frame0) delete gs_frame0;
	if (gs_frame1) delete gs_frame1;
}

void TVL1_cuda::initGpuMat(Size matSize)
{
	isFirstTime = false;

	if (d_flow) delete d_flow;
	if (d_frame0) delete d_frame0;
	if (d_frame1) delete d_frame1;
	if (gs_frame0) delete gs_frame0;
	if (gs_frame1) delete gs_frame1;

	d_frame0 = new GpuMat(matSize, CV_8UC1);
	d_frame1 = new GpuMat(matSize, CV_8UC1);
	d_flow = new GpuMat(matSize, CV_32FC2);
	gs_frame0 = new Mat(matSize, CV_8UC1);
	gs_frame1 = new Mat(matSize, CV_8UC1);
}

void TVL1_cuda::process(Mat &frame0, Mat &frame1, Mat& flow)
{
	if (isFirstTime) initGpuMat(frame1.size());

	d_frame0->upload(frame0);
	d_frame1->upload(frame1);

	tvl1->calc(*d_frame0, *d_frame1, *d_flow);

	d_flow->download(flow);
}
