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
#include "stereoodom_base.h"
#include <stdio.h>
#include <fstream>
namespace RiseOdom{
	
////////////  Normal Frame ////////////////
NormalFrame::NormalFrame(){
	this->cam_frame = TransformationMatrix();
	this->data = Frame_Data();
	this->alldata = Frame_Data();
	this->validated_data = Frame_Data();
	this->frame_type = FRAME_BAD;
	this->frame_no = -1;
	this->sec = 0.;
}
NormalFrame::NormalFrame(const NormalFrame &cpy){
	this->cam_frame = cpy.cam_frame;
	this->data = cpy.data;
	this->alldata = cpy.data;
	this->validated_data = cpy.validated_data;
	this->frame_type = cpy.frame_type;
	this->frame_no = cpy.frame_no;
	this->sec = cpy.sec;
}
NormalFrame& NormalFrame::operator=(const NormalFrame &cpy){
	this->cam_frame = cpy.cam_frame;
	this->data = cpy.data;
	this->alldata = cpy.data;
	this->validated_data = cpy.validated_data;
	this->frame_type = cpy.frame_type;
	this->frame_no = cpy.frame_no;
	this->sec = cpy.sec;
	
	return *this;
}
NormalFrame::~NormalFrame(){
	this->cam_frame = TransformationMatrix();
	this->data = Frame_Data();
	this->alldata = Frame_Data();
	this->validated_data = Frame_Data();
	this->frame_type = FRAME_BAD;
	this->frame_no = -1;
	this->sec = 0.;
}

/////////// Base Class ///////////////////
	
StereoOdom_Base::StereoOdom_Base(const int nfrmaes)
{
	// main constructor
	int nf;
	if ( (nfrmaes % 2) == 0 ) nf = nfrmaes+1;
	else nf = nfrmaes;
	this->_setframeslength(nf);
	this->_baseline = 1.;
	this->_focus = 1.;
}
StereoOdom_Base::StereoOdom_Base(const StereoOdom_Base& other)
{
	// copy constructor
	this->_setframeslength(other._frames_length);
	this->_baseline = other._baseline;
	this->_focus = other._focus;
	// frames copy
	for ( int _i = 0; _i < this->_frames_length; ++_i){
		this->frames[_i] = other.frames[_i];
	}
}
StereoOdom_Base::~StereoOdom_Base()
{
	// destructor
	this->_clean();
}

double StereoOdom_Base::_getdebugtime(const double _marktime) const
{
	double _e0 = (cv::getTickCount() - _marktime) * 1000. / cv::getTickFrequency();
	return _e0;
}
void StereoOdom_Base::_getmeanstddev(const std::vector< double >& vec, double& mean, double& stddev) const
{
	// build cv mat
	int _n = vec.size();
	cv::Mat _mat = cv::Mat(_n, 1, CV_64FC1);
	for ( int _i = 0; _i < _n; ++_i){
		_mat.at<double>(_i) = vec[_i];
	}
	cv::Mat _m, _s;
	cv::meanStdDev(_mat, _m, _s);
	mean = _m.at<double>(0);
	stddev = _s.at<double>(0);
}
void StereoOdom_Base::_getmeanstddev(const std::vector< int >& vec, double& mean, double& stddev) const
{
	// build cv mat
	int _n = vec.size();
	cv::Mat _mat = cv::Mat(_n, 1, CV_64FC1);
	for ( int _i = 0; _i < _n; ++_i){
		_mat.at<double>(_i) = vec[_i];
	}
	cv::Mat _m, _s;
	cv::meanStdDev(_mat, _m, _s);
	mean = _m.at<double>(0);
	stddev = _s.at<double>(0);
}
TransformationMatrix StereoOdom_Base::get_pose(const int frame_no) const
{
	if ( (frame_no < 0) || (frame_no >= this->_frames_length ) ){
		return this->frames.back().cam_frame;
	} else {
		return this->frames[frame_no].cam_frame;
	}
}
cv::Point2f StereoOdom_Base::get_vel() const
{
	cv::Matx61f _vels = this->get_velall();
	cv::Point2f _vel2;	_vel2.x = _vels(2);		// Z component relative to camera frame
	_vel2.y = -_vels(4);						// -Y component relative to camera frame
	return _vel2;
}
cv::Matx61f StereoOdom_Base::get_velall() const
{
	// returns a 6x1 vector 3x1 linear velocity , 3x1 angular velocity
	cv::Matx31d _oldorig, _deltatrans, _newtrans;
	_oldorig = this->frames[this->_frames_length-2].cam_frame.getTranslation();
	_newtrans = this->frames.back().cam_frame.getTranslation();
	Quaternion oldquat = this->frames[this->_frames_length-2].cam_frame.getQuaternion();
	Quaternion newquat = this->frames.back().cam_frame.getQuaternion();
	_deltatrans = oldquat.inv().RotatePoint(_newtrans - _oldorig);
	Quaternion deltaquat = oldquat.inv() * newquat;
	if ( deltaquat.getW() < 0) deltaquat.scalarmul(-1.);
	cv::Matx31d _eul = TransformationMatrix::QuatToXYZFixed(deltaquat);
	//		compute vels
	double _dt = this->frames.back().sec - this->frames[this->_frames_length-2].sec;
	_deltatrans = _deltatrans * ( 1. / _dt);
	_eul = _eul * ( 1. / _dt);
	cv::Matx61f res;
	res << _deltatrans(0), _deltatrans(1), _deltatrans(2), _eul(0), _eul(1), _eul(2);
	return res;
}
void StereoOdom_Base::initialize(const double baseline, const double focus, const cv::Matx33d cam)
{
	this->_baseline = baseline;
	this->_focus = focus;
	this->_cammatrix = cam;
}
void StereoOdom_Base::_setframeslength(const int nframes)
{
	this->_clean();
	this->_frames_length = (nframes > 2 )? nframes : 3;
	this->frames.resize(this->_frames_length);
}
void StereoOdom_Base::_wave()
{
	this->frames.push_back(NormalFrame());
	if ( this->frames.size() > this->_frames_length ) this->frames.pop_front();
}
void StereoOdom_Base::_clean()
{
	this->frames.clear();
	this->_frames_length = 0;
}
// Some other functions
int StereoOdom_Base::_getlatestkeyframe() const
{
	for ( int _i = this->_frames_length-2; _i >= 0; --_i){
		if ( this->frames[_i].frame_type == FRAME_KEYFRAME ){
			return _i;
		}
	}
	return -1;
}
int StereoOdom_Base::_getoldestkeyframe() const
{
	for ( int _i = 0; _i < this->_frames_length-1; ++_i){
		if ( this->frames[_i].frame_type ==  FRAME_KEYFRAME ){
			return _i;
		}
	}
	return -1;
}
int StereoOdom_Base::_getmiddlekeyframe() const
{
	int _k = (this->_frames_length / 2);
	if ( this->frames[_k].frame_type == FRAME_KEYFRAME ) return _k;
	return -1;
}
int StereoOdom_Base::get_frames_length() const
{
	return this->_frames_length;
}
bool StereoOdom_Base::_allframesgood() const
{
	for ( unsigned int _i = 0; _i < this->_frames_length; ++_i ){
		if ( this->frames[_i].frame_type == FRAME_BAD ) return false;
	}
	return true;
}

void StereoOdom_Base::add_imagedata(const Frame_Data& data, const double time)
{
	this->_wave();
	this->frames.back().alldata = data;
	this->frames.back().data = data.filter_invalid();
	this->frames.back().validated_data = Frame_Data();
	this->frames.back().sec = time;
	this->frames.back().frame_no = this->frames[this->_frames_length-2].frame_no + 1;
	if ( this->frames.back().data.descriptors.rows < RISEODOM_MINIMUMPTS ) this->frames.back().frame_type = FRAME_BAD;
	else {
		// 3D Points //
		int cnt3d = this->frames.back().data.points.size();
		if ( cnt3d > RISEODOM_MINIMUM3DPTS ) {
			this->frames.back().frame_type = FRAME_KEYFRAME;
		} else {
			this->frames.back().frame_type = FRAME_NORMAL;
		}
	}
}



}