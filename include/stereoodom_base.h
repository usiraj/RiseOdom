/*
 * Copyright 2015 <Usama Siraj> <osama@smme.edu.pk>
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

#ifndef STEREOODOM_BASE_H
#define STEREOODOM_BASE_H

#include "stereopreprocess.h"
#include "utils.h"

#define RISEODOM_MINIMUMPTS			20
#define RISEODOM_MINIMUM3DPTS		15

namespace RiseOdom{
	
enum FrameType { FRAME_BAD=0, FRAME_NORMAL, FRAME_KEYFRAME };

// Normal Frame
struct NormalFrame{
	int frame_no;
	double sec;
	FrameType frame_type;
	TransformationMatrix cam_frame;
	Frame_Data alldata;
	Frame_Data data;
	Frame_Data validated_data;
	////////// Constructor Destructor //////////
	NormalFrame();
	NormalFrame(const NormalFrame &cpy);
	NormalFrame& operator=(const NormalFrame &rhs);
	~NormalFrame();
};
// BASE_CLASS_ABSTRACT
class StereoOdom_Base {
public:
	// constructors
	StereoOdom_Base(const int _noframes=3);
	StereoOdom_Base(const StereoOdom_Base &other);
	virtual ~StereoOdom_Base();
	virtual void initialize(const double baseline, const double focus, const cv::Matx33d cam);
	// member functions
	TransformationMatrix get_pose(const int frame_no=-1) const;	// -1 for latest
	virtual cv::Point2f get_vel() const;
	virtual cv::Matx61f get_velall() const;
	int get_frames_length() const;
	// pure virtual members
	virtual void add_imagedata( const Frame_Data &data, const double time);
	virtual void estimate_latest_motion(cv::Mat *dbgimg = NULL) = 0;
	virtual void local_optimization() = 0;
protected:
	// utility functions
	void _wave();	// slide all data like keypts, descriptors and 3d points
	int _getlatestkeyframe() const;
	int _getoldestkeyframe() const;
	int _getmiddlekeyframe() const;
	bool _allframesgood() const;
	void _getmeanstddev(const std::vector<double> &vec, double &mean, double &stddev) const;
	void _getmeanstddev(const std::vector<int> &vec, double &mean, double &stddev) const;
	double _getdebugtime(const double _marktime) const;	// gets the time relative to marker in ms
	//members
	std::deque<NormalFrame> frames;
	cv::Matx33f _cammatrix;
	float _baseline;
	float _focus;
private:
	void _setframeslength(const int nframes);
	void _clean();
	int _frames_length;
};

	
}
#endif // STEREOODOM_BASE_H
