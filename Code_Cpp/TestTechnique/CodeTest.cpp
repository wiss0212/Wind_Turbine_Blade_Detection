#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.09;
const float NMS_THRESHOLD = 0.1;

const float CONFIDENCE_THRESHOLD = 0.25;  // Reduced to capture more possible detections

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

// Draw the predicted bounding box.
void draw_label(Mat& input_image, const string& label, int left, int top)
{
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    Point tlc(left, top);
    Point brc(left + label_size.width, top + label_size.height + baseLine);
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

vector<Mat> pre_process(Mat& input_image, Net& net)
{
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs;
}

Mat post_process(Mat input_image, vector<Mat>& outputs, const vector<string>& class_name)
{
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    if (outputs.empty() || outputs[0].empty()) {
        cerr << "Error: Output from the network is empty!" << endl;
        return input_image;
    }

    const int expected_dimensions = 8;
    const int expected_rows = 25200;
    if (outputs[0].dims != 3 || outputs[0].size[2] != expected_dimensions || outputs[0].size[1] != expected_rows) {
        cerr << "Error: Unexpected output dimensions!" << endl;
        return input_image;
    }

    float* data = (float*)outputs[0].data;

    map<int, pair<float, Rect>> best_detections;

    // Primary detection pass
    for (int i = 0; i < expected_rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float* classes_scores = data + 5;
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD)
            {
                int cls_id = class_id.x;

                if (cls_id == 0 || cls_id == 1 || cls_id == 2)
                {
                    float cx = data[0];
                    float cy = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    Rect box = Rect(left, top, width, height);

                    if (best_detections.find(cls_id) == best_detections.end() || confidence > best_detections[cls_id].first)
                    {
                        best_detections[cls_id] = { confidence, box };
                    }
                }
            }
        }
        data += expected_dimensions;
    }

    // Check if "Pale 1" was detected
    bool pale1_detected = best_detections.find(0) != best_detections.end();
    if (!pale1_detected)
    {
        // Secondary detection pass with adjusted parameters
        const float secondary_confidence_threshold = 0.10;
        const Size secondary_input_size(320, 320);  // Smaller input size

        Mat blob;
        blobFromImage(input_image, blob, 1. / 255., secondary_input_size, Scalar(), true, false);
        Net net;
        net.setInput(blob);

        vector<Mat> secondary_outputs;
        net.forward(secondary_outputs, net.getUnconnectedOutLayersNames());

        data = (float*)secondary_outputs[0].data;
        for (int i = 0; i < expected_rows; ++i)
        {
            float confidence = data[4];
            if (confidence >= secondary_confidence_threshold)
            {
                float* classes_scores = data + 5;
                Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
                Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                if (max_class_score > SCORE_THRESHOLD)
                {
                    int cls_id = class_id.x;

                    if (cls_id == 0) // Only consider "Pale 1"
                    {
                        float cx = data[0];
                        float cy = data[1];
                        float w = data[2];
                        float h = data[3];
                        int left = int((cx - 0.5 * w) * x_factor);
                        int top = int((cy - 0.5 * h) * y_factor);
                        int width = int(w * x_factor);
                        int height = int(h * y_factor);
                        Rect box = Rect(left, top, width, height);

                        if (best_detections.find(cls_id) == best_detections.end() || confidence > best_detections[cls_id].first)
                        {
                            best_detections[cls_id] = { confidence, box };
                        }
                    }
                }
            }
            data += expected_dimensions;
        }
    }

    // Draw the final bounding boxes and labels after secondary pass
    for (int cls_id = 0; cls_id < 3; ++cls_id)
    {
        if (best_detections.find(cls_id) != best_detections.end())
        {
            float confidence = best_detections[cls_id].first;
            Rect box = best_detections[cls_id].second;

            rectangle(input_image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), BLUE, 3 * THICKNESS);

            string label = format("%.2f", confidence);
            label = class_name[cls_id] + ":" + label;
            draw_label(input_image, label, box.x, box.y);
        }
        else
        {
            cerr << "Warning: Class " << class_name[cls_id] << " not detected in this frame." << endl;

            if (cls_id == 0 && best_detections.find(1) != best_detections.end() && best_detections.find(2) != best_detections.end())
            {
                Rect pale2_box = best_detections[1].second;
                Rect pale3_box = best_detections[2].second;

                int left = (pale2_box.x + pale3_box.x) / 2;
                int top = (pale2_box.y + pale3_box.y) / 2;
                int width = (pale2_box.width + pale3_box.width) / 2;
                int height = (pale2_box.height + pale3_box.height) / 2;

                Rect estimated_box(left, top, width, height);

                rectangle(input_image, Point(estimated_box.x, estimated_box.y), Point(estimated_box.x + estimated_box.width, estimated_box.y + estimated_box.height), RED, 3 * THICKNESS);

                string label = class_name[cls_id] + ": manual";
                draw_label(input_image, label, estimated_box.x, estimated_box.y);
            }
        }
    }

    return input_image;
}



int main()
{
    vector<string> class_list;
    ifstream ifs("C:/Users/daouiaouissem/Desktop/Yolo3/labels.name");
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }

    Net net;
    net = readNet("C:/Users/daouiaouissem/Desktop/Test_Technique/yolov5/runs/train/mode_yolo5/weights/best.onnx");

    string videoFile = "C:/Users/daouiaouissem/Desktop/Test_Technique/DataPart2/DataPart2/MAH02363.MP4";
    VideoCapture cap(videoFile);
    if (!cap.isOpened()) {
        cerr << "Error opening video file: " << videoFile << endl;
        return -1;
    }

    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);

    VideoWriter video("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frame_width, frame_height));

    Mat frame;
    int frameCounter = 0;  // To limit display frequency
    while (cap.read(frame)) {
        vector<Mat> detections = pre_process(frame, net);
        Mat img = post_process(frame.clone(), detections, class_list);

        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time : %.2f ms", t);
        putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

        video.write(img);

        frameCounter++;
        if (frameCounter % 10 == 0) {  // Only display every 10th frame
            imshow("Output", img);
            if (waitKey(1) >= 0) break;  // Break the loop on any key press
        }
    }

    cap.release();
    video.release();

    return 0;
}
