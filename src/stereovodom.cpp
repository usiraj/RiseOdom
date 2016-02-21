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

#include "stereovodom.h"
#include <stdio.h>
namespace RiseOdom {
/* Class Stereo Odometry
 *  
 */	
StereoVodom::StereoVodom(const int nframes) : StereoOdom_Base(nframes)
{
	// other things to initalize so the values are safe
	this->_lastokay = false;
	this->reprojerror = 0.5;
	this->setitereations(0.995, 0.5);
	this->ispkf = NULL;
	this->search_radius = 30.;
	this->loweratio = 0.9;
	this->search_threshold = 128;
	this->ransac = false;
}
StereoVodom::StereoVodom(const StereoVodom& other) : StereoOdom_Base(other)
{
	// nothing else to copy, if it is copy here
	this->_lastokay = other._lastokay;
	this->iterations = other.iterations;
	this->reprojerror = other.reprojerror;
	this->ispkf = other.ispkf;
	this->__cache_rvec = other.__cache_rvec;
	this->__cache_tvec = other.__cache_tvec;
	this->search_radius = other.search_radius;
	this->loweratio = other.loweratio;
	this->search_threshold = other.search_threshold;
	this->ransac = other.ransac;
}
StereoVodom::~StereoVodom(){
	if ( this->ispkf != NULL ){
		this->ispkf = NULL;
	}
}
void StereoVodom::setdesiredreprojectionerror(const double _err)
{
	this->reprojerror = _err;
}
void StereoVodom::setsearchcriteria(const double lowe, const int threshold, const double radius, bool rnsc)
{
	this->search_radius = radius;
	this->loweratio = lowe;
	this->search_threshold = threshold;
	this->ransac = rnsc;
}
void StereoVodom::local_optimization()
{
	NormalFrame *lastframe1 = &(this->frames[this->get_frames_length()-1]);
	NormalFrame *lastframe2 = &(this->frames[this->get_frames_length()-2]);
	double _dt2 = lastframe1->sec - lastframe2->sec;
	int  framesys = 1;
	// DO SPKF
	if ( this->dospkf ){
		if ( this->_lastokay ){
			// Measurement Update
			this->ispkf->AddObservation(lastframe2->cam_frame, lastframe1->cam_frame, _dt2, framesys, NULL);
			// store the updated result
			lastframe1->cam_frame = this->ispkf->getCameraPose();
		}
		
	}
}
void StereoVodom::setitereations(const double confidence, const double percentoutliers)
{
	double _a = log(1. - confidence);
	double _b = log(1. - pow(1. - percentoutliers, 4));
	this->iterations = int(_a / _b);
}
void StereoVodom::register_ispkf(ISPKF_2D* ptrispkf)
{
	this->ispkf = ptrispkf;
}
void StereoVodom::unregister_ispkf()
{
	this->ispkf = NULL;
}
ISPKF_2D* StereoVodom::getptr_ispkf()
{
	return this->ispkf;
}
// Important Methods
void StereoVodom::add_imagedata( const Frame_Data &data, const double time)
{
	/////////// Add Data //////////
	double _dt2 = this->frames.back().sec - this->frames[this->get_frames_length()-2].sec;
	StereoOdom_Base::add_imagedata(data, time);
	if ( this->dospkf && _dt2 < 10. ){	
	    // Predict
		this->ispkf->SigmaPrediction(_dt2);
		this->frames.back().cam_frame = this->ispkf->getCameraPose();
	} else {
		this->frames.back().cam_frame = this->frames[this->get_frames_length()-2].cam_frame;
	}
}
void StereoVodom::estimate_latest_motion(cv::Mat *dbgimg)
{
	// get latest keyframe
	int _kfr = this->_getlatestkeyframe();
	int _cfr = this->get_frames_length()-1;
	NormalFrame *curframe = &(this->frames[_cfr]);
	bool _lastst = this->_lastokay;
	this->_lastokay = false;
	if ( (_kfr < 0) || (curframe->frame_type == FRAME_BAD) ){
		this->frames.back().cam_frame = this->frames[this->get_frames_length()-2].cam_frame;
		fprintf(stderr, "carrying on transformation because of bad frame or not enough frames\n");	fflush(stdout);
	} else {
		// estimate motion using PnP between latest keyframe and current frame if it is good
		TransformationMatrix old_pi = this->frames[_kfr].cam_frame.getInverse();
		TransformationMatrix del_pose = old_pi *  this->frames.back().cam_frame;
		Frame_Data pd =	this->frames[_kfr].data.predicted_data(del_pose, this->_cammatrix, this->_baseline);
		//Frame_Data pd =	this->frames[_kfr].data;
		double srchrad;
		if ( _lastst ) {
			srchrad = this->search_radius;
		} else {
			srchrad = -1.;
		}		
		std::vector<int> mtches ;
		if ( this->ransac ){
			mtches = pd.match_with_ransac(this->frames.back().alldata, this->loweratio, this->search_threshold, srchrad,
													   this->_cammatrix, this->reprojerror, this->iterations, &(this->__cache_rvec), &(this->__cache_tvec)) ;
		} else {
			mtches = pd.match_with(this->frames.back().alldata, this->loweratio, srchrad, this->search_radius);
		}
		//// Put Valid Data ////
		int cnts = 0;
		for ( int i = 0; i < mtches.size(); ++i ){
			if ( mtches[i] != -1 ){
				cnts += 1;
			}
		}
		this->frames.back().validated_data = this->frames.back().alldata.getMatched(mtches, false).filter_invalid();
		///////// Optional Debugging //////////
		if ( dbgimg != NULL ){
			*dbgimg = RiseOdom::draw_matched_framedata(*dbgimg,  this->frames[_kfr].data, this->frames.back().alldata, mtches);
		}											   
		//////// End of Optional Debugging ////		
		if ( cnts >= 5 ){
			TransformationMatrix observed_pose;
			////////////////// Estimate Delta Motion //////////////////
			if ( this->ransac ){
				observed_pose = this->frames[_kfr].data.estimate_delta_motion(this->frames.back().alldata, mtches, this->_cammatrix,
													&(this->__cache_rvec), &(this->__cache_tvec));
			} else {
				observed_pose = this->frames[_kfr].data.estimate_delta_motion(this->frames.back().alldata, mtches, this->_cammatrix, NULL, NULL);
			}
			this->frames.back().cam_frame = this->frames[_kfr].cam_frame * observed_pose;
			this->_lastokay = true;
		} else {
			this->frames.back().cam_frame = this->frames[this->get_frames_length()-2].cam_frame;
			fprintf(stderr,"carrying on transformation because of matches less than 5\n");	fflush(stdout);
		}
	}
}
cv::Matx61f StereoVodom::get_velall() const
{
	if ( this->dospkf && this->ispkf != NULL){
		cv::Mat _x = this->ispkf->getX();
		cv::Matx61f _v;		_v(0) = _x.at<double>(7);	_v(1) = (0);	_v(2) = 0.;
		_v(3) = 0.;	_v(4) = 0.;	_v(5) = _x.at<double>(8);
		return _v;
	} else {
		return RiseOdom::StereoOdom_Base::get_velall();
	}
}
cv::Point2f StereoVodom::get_vel() const
{
	if ( this->dospkf && this->ispkf != NULL){
		cv::Mat _x = this->ispkf->getX();
		cv::Point2f _v;		_v.x = _x.at<double>(7);	_v.y = _x.at<double>(8);
		return _v;
	} else {
		return RiseOdom::StereoOdom_Base::get_vel();
	}
}

}
