// main.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;
string model_xml = "D:/projects/models/face-detection-0102/FP32/face-detection-0102.xml";
string model_bin = "D:/projects/models/face-detection-0102/FP32/face-detection-0102.bin";

std::vector<float> anchors = {
	10,13, 16,30, 33,23,
	30,61, 62,45, 59,119,
	116,90, 156,198, 373,326
};

int get_anchor_index(int scale_w, int scale_h) {
	if (scale_w == 20) {
		return 12;
	}
	if (scale_w == 40) {
		return 6;
	}
	if (scale_w == 80) {
		return 0;
	}
	return -1;
}

float get_stride(int scale_w, int scale_h) {
	if (scale_w == 20) {
		return 32.0;
	}
	if (scale_w == 40) {
		return 16.0;
	}
	if (scale_w == 80) {
		return 8.0;
	}
	return -1;
}

float sigmoid_function(float a)
{
	float b = 1. / (1. + exp(-a));
	return b;
}

void face_detection_demo();
void yolov5_onnx_demo();
void object_segmentation();
int main(int argc, char** argv) {
	yolov5_onnx_demo();
}

void yolov5_onnx_demo() {
	Mat src = imread("E:/Project/cpp_project/VideoCutout/VideoCutout/000000000552.jpg");
	int image_height = src.rows;
	int image_width = src.cols;
	VideoCapture cap;
	cap.open("E:/Project/cpp_project/VideoCutout/VideoCutout/TEST_01.mp4");


	// 创建IE插件, 查询支持硬件设备
	Core ie;
	vector<string> availableDevices = ie.GetAvailableDevices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
	}

	//  加载检测模型
	auto network = ie.ReadNetwork("E:/Project/cpp_project/VideoCutout/VideoCutout/deeplab_ghostnet.xml", "E:/Project/cpp_project/VideoCutout/VideoCutout/deeplab_ghostnet.bin");
	// auto network = ie.ReadNetwork("D:/python/yolov5/yolov5s.onnx");

	// 请求网络输入与输出信息
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());
	// 设置输入格式
	for (auto& item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::FP32);
		input_data->setLayout(Layout::NCHW);
		input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
		input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
	}

	// 设置输出格式
	for (auto& item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
	}
	auto executable_network = ie.LoadNetwork(network, "CPU");

	// 处理解析输出结果
	vector<Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;
	vector<int> result;
	

	// 请求推断图
	auto infer_request = executable_network.CreateInferRequest();
	float scale_x = image_width / 640.0;
	float scale_y = image_height / 640.0;

	int64 start = getTickCount();
	int frame_num = 0;
	/** Iterating over all input blobs **/
	for (auto& item : input_info) {
		auto input_name = item.first;

		/** Getting input blob **/
		auto input = infer_request.GetBlob(input_name);
		size_t num_channels = input->getTensorDesc().getDims()[1];
		size_t h = input->getTensorDesc().getDims()[2];
		size_t w = input->getTensorDesc().getDims()[3];
		size_t image_size = h * w;
		Mat blob_image;

		// 解析视频帧
		Mat frame;
		for (;;) {
			cap >> frame;
			if (frame.empty()) break;
			if (frame.rows == 0) break;
			resize(frame, blob_image, Size(w, h));
			cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

			// NCHW
			float* data = static_cast<float*>(input->buffer());
			for (size_t row = 0; row < h; row++) {
				for (size_t col = 0; col < w; col++) {
					for (size_t ch = 0; ch < num_channels; ch++) {
						data[image_size * ch + row * w + col] = float(blob_image.at<Vec3b>(row, col)[ch]) / 255.0;
					}
				}
			}

			// 执行预测
			infer_request.Infer();

			//int mask[56][56];
			//int mask[448][448];
			Mat mask = Mat::zeros(448, 448, CV_8UC1);
			auto& mask_data = mask.data;
			for (auto& item : output_info) {
				auto output_name = item.first;
				//printf("output_name : %s \n", output_name.c_str());
				// 获取输出数据
				auto output = infer_request.GetBlob(output_name);

				const float* output_blob = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
				const SizeVector outputDims = output->getTensorDesc().getDims();
				const int out_n = outputDims[0];
				const int out_c = outputDims[1];
				const int side_h = outputDims[2];
				const int side_w = outputDims[3];
				//const int side_data = outputDims[4];
				const int side_data = 1;
				float stride = get_stride(side_h, side_h);
				int anchor_index = get_anchor_index(side_h, side_h);
				//printf("number of images: %d, channels : %d, height: %d, width : %d, out_data:%d \n", out_n, out_c, side_h, side_w, side_data);
				int side_square = side_h * side_w;
				int side_data_square = side_square * side_data;
				int side_data_w = side_w * side_data;
				for (int i = 0; i < side_square; ++i) {
					for (int c = 0; c < out_c - 1; c++) {
						int row = i / side_h;
						int col = i % side_h;
						int background_index = c * side_data_square + row * side_data_w + col * side_data;
						int object_index = (c + 1) * side_data_square + row * side_data_w + col * side_data;

						// 人像判断
						if (output_blob[object_index] > output_blob[background_index]) {
							//mask_data[background_index] = 1;
							mask.at<uchar>(row, col) = 255;
						}
						

					}
				}
			}
			frame_num += 1;
			//cv::imshow("例子3", mask);
			//if (cv::waitKey(33) >= 0)
			//	break;

		}

		
		
	}

	

	

	//vector<int> indices;
	//NMSBoxes(boxes, confidences, 0.25, 0.5, indices);
	//for (size_t i = 0; i < indices.size(); ++i)
	//{
	//	int idx = indices[i];
	//	Rect box = boxes[idx];
	//	rectangle(src, box, Scalar(140, 199, 0), 4, 8, 0);
	//}
	float fps = getTickFrequency() / (getTickCount() - start);
	float time = (getTickCount() - start) / getTickFrequency();

	ostringstream ss;
	ss << "FPS : " << fps << " segmentation time: " << time * 1000 << " ms";
	printf("number of images: %d, FPS : %f, time: %f \n", frame_num, fps, time*1000);
	//putText(src, ss.str(), Point(20, 50), 0, 1.0, Scalar(0, 0, 255), 2);

	//imshow("OpenVINO2021R2+YOLOv5对象检测", src);
	//imwrite("E:/Project/cpp_project/VideoCutout/VideoCutout/openvino2021_yolov5_test.png", src);
	//waitKey(0);
}

void face_detection_demo() {
	Mat src = imread("D:/images/persons.png");
	int image_height = src.rows;
	int image_width = src.cols;

	// 创建IE插件, 查询支持硬件设备
	Core ie;
	vector<string> availableDevices = ie.GetAvailableDevices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
	}

	//  加载检测模型
	auto network = ie.ReadNetwork(model_xml, model_bin);

	// 请求网络输入与输出信息
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());
	// 设置输入格式
	for (auto& item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::U8);
		input_data->setLayout(Layout::NCHW);
		input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
		input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
	}
	printf("get it \n");

	// 设置输出格式
	for (auto& item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
	}

	// 创建可执行网络对象
	// ie.AddExtension(std::make_shared<Extension>("C:/Intel/openvino_2020.1.033/deployment_tools/ngraph/lib/ngraph.dll"), "CPU");
	auto executable_network = ie.LoadNetwork(network, "CPU");
	// auto executable_network = ie.LoadNetwork(network, "MYRIAD");

	// 请求推断图
	auto infer_request = executable_network.CreateInferRequest();

	/** Iterating over all input blobs **/
	for (auto& item : input_info) {
		auto input_name = item.first;

		/** Getting input blob **/
		auto input = infer_request.GetBlob(input_name);
		size_t num_channels = input->getTensorDesc().getDims()[1];
		size_t h = input->getTensorDesc().getDims()[2];
		size_t w = input->getTensorDesc().getDims()[3];
		size_t image_size = h * w;
		Mat blob_image;
		resize(src, blob_image, Size(w, h));

		// NCHW
		unsigned char* data = static_cast<unsigned char*>(input->buffer());
		for (size_t row = 0; row < h; row++) {
			for (size_t col = 0; col < w; col++) {
				for (size_t ch = 0; ch < num_channels; ch++) {
					data[image_size * ch + row * w + col] = blob_image.at<Vec3b>(row, col)[ch];
				}
			}
		}
	}

	// 执行预测
	infer_request.Infer();

	// 处理输出结果
	for (auto& item : output_info) {
		auto output_name = item.first;

		// 获取输出数据
		auto output = infer_request.GetBlob(output_name);
		const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
		const SizeVector outputDims = output->getTensorDesc().getDims();
		const int maxProposalCount = outputDims[2];
		const int objectSize = outputDims[3];

		// 解析输出结果
		for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
			float label = detection[curProposal * objectSize + 1];
			float confidence = detection[curProposal * objectSize + 2];
			float xmin = detection[curProposal * objectSize + 3] * image_width;
			float ymin = detection[curProposal * objectSize + 4] * image_height;
			float xmax = detection[curProposal * objectSize + 5] * image_width;
			float ymax = detection[curProposal * objectSize + 6] * image_height;
			if (confidence > 0.5) {
				printf("label id : %d\n", static_cast<int>(label));
				Rect rect;
				rect.x = static_cast<int>(xmin);
				rect.y = static_cast<int>(ymin);
				rect.width = static_cast<int>(xmax - xmin);
				rect.height = static_cast<int>(ymax - ymin);
				putText(src, "OpenVINO-2021R02", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2, 8);
				rectangle(src, rect, Scalar(0, 255, 255), 2, 8, 0);
			}
			std::cout << std::endl;
		}
	}
	imshow("openvino+ssd人脸检测", src);
	imwrite("D:/result.png", src);
	waitKey(0);
	return;
}

void object_segmentation() {

}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
