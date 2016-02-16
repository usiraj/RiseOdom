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

// add realtime and simtime

#ifndef STEREOVODOM2_H
#define STEREOVODOM2_H

#define STEREOVODOM2_MINIMUMPTS		20
#define STEREOVODOM2_MINIMUM3DPTS	15
#define STEREOVODOM2_MTNESTRATIOL	0.85
#define STEREOVODOM2_BALOWE			0.65

#include "stereoodom_base.h"
#include "ispkf_motionmodels.h"

namespace RiseOdom{

class StereoVodom: public StereoOdom_Base
{
public:
	StereoVodom(const int nframes=3);
	StereoVodom(const StereoVodom& other);
	virtual ~StereoVodom();
	virtual void add_imagedata( const Frame_Data &data, const double time);
	virtual void estimate_latest_motion(cv::Mat *dbimg = NULL);
	virtual void local_optimization();	// do the observation update step
	virtual cv::Point2f get_vel() const;
	virtual cv::Matx61f get_velall() const;
	void EnableSPKF(bool status){ this->dospkf = status ; }
	// some members
	void setdesiredreprojectionerror(const double _err=4);
	void setitereations(const double confidence=0.95, const double percentoutliers=0.6);
	void setsearchcriteria(const double lowe=0.9, const int threshold = 128, const double radius = 30., bool ransac=true);
	// register ispkf
	void register_ispkf(ISPKF_2D *ptrispkf);	// initialized  ispkf
	void unregister_ispkf();
	ISPKF_2D* getptr_ispkf();
protected:
	int iterations;
	double reprojerror;
	double loweratio;
	double search_radius;
	int search_threshold;
	bool ransac;	
	// ispkf
	ISPKF_2D *ispkf;	
private:
	bool _lastokay;
	bool dospkf;
	
	cv::Mat __cache_rvec;	// updated after ransac motion filtering
	cv::Mat __cache_tvec;	// updated after ransac motion filtering
};


}
#endif // STEREOVODOM2_H
