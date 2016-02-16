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

#ifndef ISPKF_MOTIONMODELS_H
#define ISPKF_MOTIONMODELS_H

#include "spkf.h"
#include "transformation_matrix.h"

/*
 * Base class for motion models 
 */

class ISPKF_MM : public SPKF {
public:
ISPKF_MM(const ISPKF_MM& other);
ISPKF_MM(const unsigned int dimX, const double _alpha = 1E-4, const double _beta = 2, const double _k = 0);
virtual ~ISPKF_MM();
// setting and getting transform
void setBaseCameraTransform(const TransformationMatrix &trmat);
TransformationMatrix getBaseCameraTransform() const;
// setting and getting pose
void setBasePose(const cv::Matx31d &globX, const cv::Matx31d &globAng);
void setBasePose(const TransformationMatrix &pose);
void setCameraPose(const TransformationMatrix &pose);
TransformationMatrix getBasePose() const;
TransformationMatrix getCameraPose() const;
// start filter
virtual void StartFilter() = 0;		// make this class abstract
virtual void AddObservation(const TransformationMatrix &prev, const TransformationMatrix &cur, const double _dt , const int flag, void *ptr_data) = 0;	// to call observation
static cv::Mat SSTmat(const TransformationMatrix &tmat);
// computes 3x1 linear vel and 3x1 angular vel in before frame
static cv::Mat ComputeVels(const TransformationMatrix &before, const TransformationMatrix &after, const double _dt);
static TransformationMatrix TMatSS(const cv::Mat& svec);
// whether initialized or not
bool isXinitialized() const { return this->_initialized;}
void setXinitialized(const bool isdo){ this->_initialized = isdo;}
protected:	
	TransformationMatrix bTcam;
	bool _initialized;
private:
	
};


class ISPKF_2D : public ISPKF_MM {
	/*
	* Body Frame System Model
	*	A 6DOF Global robot pose with Vforw and Vang;
	* 	A 9*1 vector	position , quaternion
	* 	States are global : [tx,ty,tz,qw, qx, qy, qz], local : [velx, velangz]
	*/
public:
ISPKF_2D(const double _alpha = 1E-4, const double _beta = 2, const double _k = 0);
ISPKF_2D(const ISPKF_2D& other);
virtual ~ISPKF_2D();
// some public members
virtual void StartFilter();
// flag is 0 zero for robot frame
// flag is 1 for camera frame
virtual void AddObservation(const TransformationMatrix &prev, const TransformationMatrix &cur, const double _dt , const int flag, void *ptr_data);
// flag is one for
void setstatecovs(const double pos=0.001, const double ang=0.01, const double angaxis=0.0001 , const double velfrw=0.1, const double velang=0.01);
void setrmatcovs(const double pos=0.00001, const double angs=0.000001, const double angaxis=0.0001, const double velfrw=0.1, const double velang=0.01);
// static models
static cv::Mat ObsvGlobalPosition(const cv::Mat &inp, void *ptr, const cv::Mat noise=cv::Mat());
static cv::Mat ObsvGlobalQuaternion(const cv::Mat &inp, void *ptr, const cv::Mat noise=cv::Mat());
static cv::Mat ObsvLocalPlanarVels(const cv::Mat &inp, void *ptr, const cv::Mat noise=cv::Mat());
static cv::Mat CVMotionModel(const cv::Mat &inp, const double _dt, void *ptr, const cv::Mat noise=cv::Mat());//
cv::Mat getRpos() const { return this->_Rpos.clone(); }
cv::Mat getRqtr() const { return this->_Rqtr.clone(); }
cv::Mat getRplanarvel() const { return this->_Rplanarvel.clone(); }
protected:
	cv::Mat _Rpos;
	cv::Mat _Rqtr;
	cv::Mat _Rplanarvel;
	virtual cv::Mat _correct_constraints(const cv::Mat &mat) const;
private:
	
};


#endif // ISPKF_MOTIONMODELS_H
