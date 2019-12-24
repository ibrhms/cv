#include "brox_cuda.h"

brox_cuda::brox_cuda()
{
	brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
	isFirstTime = true;
}

brox_cuda::~brox_cuda()
{
	if (d_flow) delete d_flow;
	if (d_frame0) delete d_frame0;
	if (d_frame1) delete d_frame1;
	if (d_frame0f) delete d_frame0f;
	if (d_frame1f) delete d_frame1f;
	if (gs_frame0) delete gs_frame0;
	if (gs_frame1) delete gs_frame1;
}

void brox_cuda::initGpuMat(Size matSize)
{
	isFirstTime = false;

	if (d_flow) delete d_flow;
	if (d_frame0) delete d_frame0;
	if (d_frame1) delete d_frame1;
	if (d_frame0f) delete d_frame0f;
	if (d_frame1f) delete d_frame1f;
	if (gs_frame0) delete gs_frame0;
	if (gs_frame1) delete gs_frame1;

	d_frame0 = new GpuMat(matSize, CV_8UC1);
	d_frame1 = new GpuMat(matSize, CV_8UC1);
	d_frame0f = new GpuMat(matSize, CV_32FC1);
	d_frame1f = new GpuMat(matSize, CV_32FC1);
	d_flow = new GpuMat(matSize, CV_32FC2);
	gs_frame0 = new Mat(matSize, CV_8UC1);
	gs_frame1 = new Mat(matSize, CV_8UC1);
}

void brox_cuda::process(Mat &frame0, Mat &frame1, Mat& flow)
{
	if (isFirstTime) initGpuMat(frame1.size());

	d_frame0->upload(frame0);
	d_frame1->upload(frame1);

	d_frame0->convertTo(*d_frame0f, CV_32F, 1.0 / 255.0);
	d_frame1->convertTo(*d_frame1f, CV_32F, 1.0 / 255.0);

	brox->calc(*d_frame0f, *d_frame1f, *d_flow);

	d_flow->download(flow);
}
