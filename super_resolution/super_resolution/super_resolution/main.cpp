#include <iostream>
#include <iomanip>
#include <string>
#include <ctype.h>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"
#include "opencv2/opencv_modules.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;
//
//int main(int argc, char*argv[])
//{
//
//	String my_file = "D://Documents//programming//bram_playground//super_resolution//super_resolution//x64//Debug//test.mp4";
//	std::cout << "Video File " << my_file << std::endl;
//	cv::VideoCapture input_video;
//
//	if (input_video.open(my_file))
//	{
//		std::cout << "Video file open " << std::endl;
//	}
//	else
//	{
//		std::cout << "Not able to Video file open " << std::endl;
//
//	}
//	namedWindow("My_Win", 1);
//	namedWindow("Segemented", 1);
//	Mat cap_img;
//	for (;;)
//	{
//		input_video >> cap_img;
//		imshow("My_Win", cap_img);
//		waitKey(0);
//	}
//	return 0;
//}

//int main(void)
//{
//	std::cout << cv::getBuildInformation() << std::endl;
//	std::cout << cv::checkHardwareSupport(CV_CPU_SSE4_2);
//}

#define MEASURE_TIME(op) \
    { \
        TickMeter tm; \
        tm.start(); \
        op; \
        tm.stop(); \
        cout << tm.getTimeSec() << " sec" << endl; \
    }

static Ptr<cv::superres::DenseOpticalFlowExt> createOptFlow(const string& name, bool useGpu)
{
	if (name == "farneback")
	{
		if (useGpu)
			return cv::superres::createOptFlow_Farneback_CUDA();
		else
			return cv::superres::createOptFlow_Farneback();
	}
	/*else if (name == "simple")
		return createOptFlow_Simple();*/
	else if (name == "tvl1")
	{
		if (useGpu)
			return cv::superres::createOptFlow_DualTVL1_CUDA();
		else
			return cv::superres::createOptFlow_DualTVL1();
	}
	else if (name == "brox")
		return cv::superres::createOptFlow_Brox_CUDA();
	else if (name == "pyrlk")
		return cv::superres::createOptFlow_PyrLK_CUDA();
	else
		cerr << "Incorrect Optical Flow algorithm - " << name << endl;

	return Ptr<cv::superres::DenseOpticalFlowExt>();
}

int main(int argc, const char* argv[])
{
	CommandLineParser cmd(argc, argv,
		"{ v video      |           | Input video (mandatory)}"
		"{ o output     |           | Output video }"
		"{ s scale      | 2         | Scale factor }"
		"{ i iterations | 200       | Iteration count }"
		"{ t temporal   | 4        | Radius of the temporal search area }"
		"{ f flow       | farneback | Optical flow algorithm (farneback, tvl1, brox, pyrlk) }"
		"{ g gpu        | CUDA     | CPU as default device, cuda for CUDA }"
		"{ h help       | false     | Print help message }"
	);

	//const string inputVideoName = cmd.get<string>("video");
	const string inputVideoName = "D://Documents//programming//bram_playground//super_resolution//super_resolution//x64//Debug//2.avi";
	if (cmd.get<bool>("help") || inputVideoName.empty())
	{
		cout << "This sample demonstrates Super Resolution algorithms for video sequence" << endl;
		cmd.printMessage();
		return EXIT_SUCCESS;
	}

	//const string outputVideoName = cmd.get<string>("output");
	const string outputVideoName = "D://Documents//programming//bram_playground//super_resolution//super_resolution//x64//Debug//result.mp4";
	const int scale = cmd.get<int>("scale");
	const int iterations = cmd.get<int>("iterations");
	const int temporalAreaRadius = cmd.get<int>("temporal");
	const string optFlow = cmd.get<string>("flow");
	string gpuOption = cmd.get<string>("gpu");

	std::transform(gpuOption.begin(), gpuOption.end(), gpuOption.begin(), ::tolower);

	bool useCuda = gpuOption.compare("cuda") == 0;
	Ptr<SuperResolution> superRes;

	if (useCuda)
		superRes = createSuperResolution_BTVL1_CUDA();
	else
		superRes = createSuperResolution_BTVL1();

	Ptr<cv::superres::DenseOpticalFlowExt> of = createOptFlow(optFlow, useCuda);

	if (of.empty())
		return EXIT_FAILURE;
	superRes->setOpticalFlow(of);

	superRes->setScale(scale);
	superRes->setIterations(iterations);
	superRes->setTemporalAreaRadius(temporalAreaRadius);

	Ptr<FrameSource> frameSource;
	if (useCuda)
	{
		// Try to use gpu Video Decoding
		try
		{
			frameSource = createFrameSource_Video_CUDA(inputVideoName);
			Mat frame;
			frameSource->nextFrame(frame);
		}
		catch (const cv::Exception&)
		{
			frameSource.release();
		}
	}
	if (!frameSource)
		frameSource = createFrameSource_Video(inputVideoName);

	// skip first frame, it is usually corrupted
	{
		Mat frame;
		frameSource->nextFrame(frame);
		cout << "Input           : " << inputVideoName << " " << frame.size() << endl;
		cout << "Scale factor    : " << scale << endl;
		cout << "Iterations      : " << iterations << endl;
		cout << "Temporal radius : " << temporalAreaRadius << endl;
		cout << "Optical Flow    : " << optFlow << endl;
		cout << "Mode            : " << (useCuda ? "CUDA" : "CPU") << endl;
	}

	superRes->setInput(frameSource);

	VideoWriter writer;

	int counter = 0;
	for (int i = 0;; ++i)
	{
		cout << '[' << setw(3) << i << "] : " << flush;
		Mat result;

		MEASURE_TIME(superRes->nextFrame(result));

		if (result.empty())
			break;

		//imshow("Super Resolution", result);

		/*if (waitKey(1000) > 0)
			break;*/

		if (!outputVideoName.empty())
		{
			if (!writer.isOpened())
				writer.open(outputVideoName, VideoWriter::fourcc('X', 'V', 'I', 'D'), 25.0, result.size());
			writer << result;
		}

		counter++;
		if (counter >= 10)
			break;
	}

	return 0;
}