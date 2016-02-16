/*
 * Copyright 2016 <Usama Siraj> <osama@smme.edu.pk>
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

#ifndef STEREOPREPROCESS_H
#define STEREOPREPROCESS_H

#include <opencv2/opencv.hpp>
#include "transformation_matrix.h"

#define IMGPREPROCESS_DEFAULT_YERR 1
#define IMGPREPROCESS_DEFAULT_XMAX 32
#define IMGPREPROCESS_DEFAULT_LOWE 0.6
#define IMGRPEPROCESS_DEFAULT_PARSCORE 64
#define IMGPREPROCESS_DEFAULT_BUCKET 30
#define IMGPREPROCESS_DEFAULT_MAXPTS 2000
#define IMGPREPROCESS_DEFAULT_THRESHOLD 8

namespace RiseOdom{

struct Stereo_Image_Points{
	cv::Point2f left;
	cv::Point2f right;	
	cv::Point3f leftcamPoint;
	bool invalid;
	void updateCamPointHorizontal(const cv::Matx33f &cam, const float _baseline, const float _mindisp=4.);
	void updateCamPointVertical(const cv::Matx33f &cam, const float _baseline, const float _mindisp=4.);
	/////////////// Constructors / Destructors /////////////////
	Stereo_Image_Points();
	Stereo_Image_Points(const Stereo_Image_Points &other);
	Stereo_Image_Points& operator=(const Stereo_Image_Points &other);
	~Stereo_Image_Points();
	void print_screen() const;
};	
	
struct Frame_Data{
	int frame_id;
	std::vector<Stereo_Image_Points> points;
	cv::Mat descriptors;
	////////////// Constructors / Destructors //////////////////
	Frame_Data();
	Frame_Data(const Frame_Data &other);
	Frame_Data &operator=(const Frame_Data &other);
	~Frame_Data();
	//////////// Helper Functions //////////////////////////////
	Frame_Data filter_invalid() const;
	void predict(const TransformationMatrix &trmat, const cv::Matx33f &cam, const float _baseline);
	Frame_Data predicted_data(const TransformationMatrix &trmat, const cv::Matx33f &cam, const float _baseline) const;
	void print_screen() const;
	cv::Mat getPts2D() const;
	cv::Mat getPts3D() const;
	int getValidCount() const;
	//////////// Matchers //////////////////////////////////////
	cv::Mat neighborhood_matrix(const Frame_Data &other, const float _maxradius) const;	
	std::vector<int> match_with(const Frame_Data &other, const float loweratio,
								const int parscore ,const float maxradius) const;	// -1 for match failure
	std::vector<int> match_with_ransac(const Frame_Data &other, const float loweratio, const int parscore, const float maxradius,
									   const cv::Matx33f &cam, const float reprojerrors, const int max_iterations, cv::Mat *rvec = NULL, cv::Mat *tvec = NULL) const;
	Frame_Data getMatched(const std::vector<int> &mtchs, bool islhs ) const;
	TransformationMatrix estimate_delta_motion(const Frame_Data &other, const std::vector<int> &match_list,
											   const cv::Matx33f &trmat, cv::Mat *rvec = NULL, cv::Mat *tvec = NULL) const;	
};

class StereoPreProcess
{
public:
	/////////////// Constructors / Destructors ///////////////
	StereoPreProcess();
	StereoPreProcess(const StereoPreProcess& other);
	~StereoPreProcess();
	StereoPreProcess& operator=(const StereoPreProcess& other);
	// initialize members
	void initialize(const int bwin, const int maxpts, const float th, const cv::Size imgsize);
	void setMatching(const int y, const int x, const float lowe, const int par_score);
	void setCam(const cv::Matx33f &camera, const float baseline, const float mindisparity);
	////////////// Get / Set Members /////////////////////////
	cv::Matx33f getCamIntrinsic() const;
	float getBaseline() const;
	float getMinimumDisparity() const;
	Frame_Data getFrameData() const;
	///////////// Process Image //////////////////////////////
	void ProcessImages(const cv::Mat &imgl, const cv::Mat &imgr);
protected:
	// camera properties
	cv::Matx33f cam;
	float baseline;
	float mindisp;
	// fast feature detector propoerties
	int bucketwindow;
	int maxpoints;
	float threshold;
	// stereo maching properties
	int _yerr;
	int _xmax;
	float loweratio;
	int parscore;
	// detectors
	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> fe;
	/////////////// current frame data /////////////
	Frame_Data data;
private:
	cv::Mat horizontal_scan_neighborhoodMatrix(const std::vector<cv::KeyPoint> &kpl, const std::vector<cv::KeyPoint> &kpr) const;
};

//////////////////////////////// Public Helper Functions ///////////////////////////////
cv::Point2f leftcam2imageplane(const cv::Matx33f &cam, const cv::Point3f &cam_point);
cv::Point2f frameN2imageplane(const cv::Matx33f &cam, const TransformationMatrix &Cam2FrameN, const cv::Point3f &frameN_point);
Stereo_Image_Points cam2stereoimageplane(const cv::Matx33f &cam, const float _baseline, const cv::Point3f &cam_point,
										 bool horizontal=true);
Stereo_Image_Points frameN2stereoimageplane(const cv::Matx33f &cam, const float _baseline,
								const TransformationMatrix &Cam2FrameN, const cv::Point3f &frameN_point, bool horizontal=true);
cv::Mat draw_matched_framedata(const cv::Mat &img, const Frame_Data &fd1, const Frame_Data &fd2,
							   const std::vector<int> &mtchs );
cv::Mat draw_framedata(const cv::Mat &img, const Frame_Data &fd, int clr_scheme=1, bool numbering=false);
cv::Mat draw_framescanmatch(const cv::Mat &imgl, const cv::Mat &imgr, const Frame_Data &fd);
cv::Mat draw_transforminfo(const cv::Mat &img, const TransformationMatrix &trmat, const double frwdvel, const double angvel);

}

#endif // STEREOPREPROCESS_H
