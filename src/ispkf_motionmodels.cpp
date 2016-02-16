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
 */

#include "ispkf_motionmodels.h"
#include "transformation_matrix.h"
#include "utils.h"
#include <stdio.h>


/*
 * ISPKF_2D CLASS
 */
ISPKF_2D::ISPKF_2D(const ISPKF_2D& other): ISPKF_MM(other)
{
	// members of this class
	this->_Rpos = other._Rpos;
	this->_Rqtr = other._Rqtr;
	this->_Rplanarvel = other._Rplanarvel;
	this->register_transition_func(ISPKF_2D::CVMotionModel);
}
ISPKF_2D::ISPKF_2D(const double _alpha, const double _beta, const double _k): ISPKF_MM( 9 , _alpha, _beta, _k)
{
	// members of this class
	this->setstatecovs();
	this->setrmatcovs();
	this->register_transition_func(ISPKF_2D::CVMotionModel);
}
ISPKF_2D::~ISPKF_2D()
{ }
void ISPKF_2D::StartFilter()
{
	// set appropriate transition function
	this->unregister_obsrvall();
	// re-register functions
	this->register_obsrvfunc(ISPKF_2D::ObsvGlobalPosition, this->_Rpos);
	this->register_obsrvfunc(ISPKF_2D::ObsvGlobalQuaternion, this->_Rqtr);
	this->register_obsrvfunc(ISPKF_2D::ObsvLocalPlanarVels, this->_Rplanarvel);
	this->register_transition_func(ISPKF_2D::CVMotionModel);
	// start now
	this->Start();
}
void ISPKF_2D::AddObservation(const TransformationMatrix &prev, const TransformationMatrix &cur, const double _dt ,
							  const int flag, void *ptr_data)
{
	// call the observation function of base class
	cv::Mat _zvels, _zcurs;
	if ( flag == 0 ){
		// tmat from base frame
		_zvels = ISPKF_MM::ComputeVels(prev, cur, _dt);
		_zcurs = ISPKF_MM::SSTmat(cur);
	} else if ( flag == 1) {
		// tmat from cam frame
		TransformationMatrix bmat_prev = this->bTcam * prev * this->bTcam.getInverse();
		TransformationMatrix bmat_cur = this->bTcam * cur * this->bTcam.getInverse();
		_zvels = ISPKF_MM::ComputeVels(bmat_prev, bmat_cur, _dt);
		_zcurs = ISPKF_MM::SSTmat(bmat_cur);		
	} else {
		CV_Error(CV_StsBadArg, "flag should be either 1 or 0" );
	}
	if ( _zvels.at<double>(0) > MAXIMUM_POSSIBLE_VEL || _zvels.at<double>(0) < -MAXIMUM_POSSIBLE_VEL ){
		fprintf(stderr, "Ignoring Observation wrong data\n");
		return;
	}	
	cv::Mat _obspv, _obsgp, _obsgq;
	_obsgp = _zcurs(cv::Rect(0, 0, 1, 3));
	_obsgq = _zcurs(cv::Rect(0, 3, 1, 4));
	_obspv = cv::Mat(2,1, CV_64FC1);	_obspv.at<double>(0) = _zvels.at<double>(0);
	_obspv.at<double>(1) = _zvels.at<double>(5);
	
	if ( !(this->_initialized) ){
		for ( int _i = 0 ; _i < _zcurs.rows; ++_i ){
			this->_X.at<double>(_i) = _zcurs.at<double>(_i);
		}
		this->_X.at<double>(7) = _obspv.at<double>(0);		this->_X.at<double>(8) = _obspv.at<double>(1);
		this->setXinitialized(true);
	} else {
		this->MeasurementUpdate(_obspv, 2, ptr_data);
		// this->MeasurementUpdate(_obsgq, 1, ptr_data);
		// this->MeasurementUpdate(_obsgp, 0, ptr_data);
	}
}
void ISPKF_2D::setstatecovs(const double pos, const double ang,const double angaxis , const double velfrw, const double velang)
{
	cv::Mat _diag(9, 1, CV_64FC1);
	for ( int _i = 0; _i < 3; ++_i ){
		_diag.at<double>(_i) = pos;
		_diag.at<double>(_i + 4) = ang;
	}
	_diag.at<double>(3) = angaxis;	
	_diag.at<double>(7) = velfrw;
	_diag.at<double>(8) = velang;
	this->setQDiagonal(_diag);
}
void ISPKF_2D::setrmatcovs(const double pos, const double angs, const double angaxis,  const double velfrw, const double velang)
{
	//
	this->_Rpos = cv::Mat::zeros(3,3, CV_64FC1);
	this->_Rqtr = cv::Mat::zeros(4,4, CV_64FC1);
	this->_Rplanarvel = cv::Mat::zeros(2,2, CV_64FC1);
	this->_Rpos.at<double>(0,0) = pos;	this->_Rpos.at<double>(1,1) = pos;	this->_Rpos.at<double>(2,2) = pos;
	this->_Rqtr.at<double>(0,0) = angaxis;
	this->_Rqtr.at<double>(1,1) = angs;	this->_Rqtr.at<double>(2,2) = angs;	this->_Rqtr.at<double>(3,3) = angs;
	this->_Rplanarvel.at<double>(0,0) = velfrw;		this->_Rplanarvel.at<double>(1,1) = velang;
}
cv::Mat ISPKF_2D::CVMotionModel(const cv::Mat& input, const double _dt, void* ptr, const cv::Mat noise)
{
	// Okay with quaternions
	CV_Assert(input.rows >= 9);
	CV_Assert(input.cols == 1);
	CV_Assert(input.type() == CV_64FC1);
	cv::Mat res(input.rows, 1, CV_64FC1);
	cv::Mat noisyiput;
	if ( !(noise.empty() ) ){
		CV_Assert(input.rows == noise.rows);
		noisyiput = input + noise;
	} else {
		noisyiput = input;
	}
	
	// trying quaternion for transformation
	Quaternion oldquat(noisyiput.at<double>(3), noisyiput.at<double>(4), noisyiput.at<double>(5), noisyiput.at<double>(6));
	cv::Matx31d _oldorig, _deltatrans, _newtrans;
	_oldorig << noisyiput.at<double>(0), noisyiput.at<double>(1), noisyiput.at<double>(2);
	_deltatrans << 0., 0., 0.;
	// compute the delta change
	_deltatrans(0) = noisyiput.at<double>(7) * _dt ; //+ 0.5 * ( _dt * _dt * inp.at<double>(9) );
	double _deltaang = noisyiput.at<double>(8) * _dt ; //+ 0.5 * ( _dt * _dt * inp.at<double>(10) );
	Quaternion _deltaq(cos(_deltaang/ 2.), 0., 0., sin(_deltaang / 2.));		// around z axis
	// update the delta change
	Quaternion _newq = oldquat * _deltaq;
	if ( _newq.getW() < 0) _newq.scalarmul(-1.);
	_newtrans = oldquat.RotatePoint(_deltatrans);
	_newtrans = _oldorig + _newtrans;
	// copy the newly calculated position
	res.at<double>(0) = _newtrans(0);		res.at<double>(1) = _newtrans(1);	res.at<double>(2) = _newtrans(2);
	res.at<double>(3) = _newq.getW();	res.at<double>(4) = _newq.getX();	res.at<double>(5) = _newq.getY();	res.at<double>(6) = _newq.getZ();
	// copy velocities to maintain them
	res.at<double>(7) = noisyiput.at<double>(7); // - _dt * inp.at<double>(9);
	res.at<double>(8) = noisyiput.at<double>(8); // - _dt * inp.at<double>(10);
	// copy remainder
	for ( int _i = 9; _i < noisyiput.rows; ++_i ) res.at<double>(_i) = noisyiput.at<double>(_i);
	//res.at<double>(9) = inp.at<double>(9) * exp( - 100*_dt);
	//res.at<double>(10) = inp.at<double>(10) * exp ( -10*_dt);	// decaying acceleartion assumption
	return res;	
}
cv::Mat ISPKF_2D::ObsvGlobalPosition(const cv::Mat& inp, void* ptr, const cv::Mat noise)
{
	CV_Assert(inp.rows >= 9);
	CV_Assert(inp.cols == 1);
	CV_Assert(inp.type() == CV_64FC1);
	cv::Mat res(3, 1, CV_64FC1);
	for ( int _i = 0; _i < 3; ++_i ) res.at<double>(_i) = inp.at<double>(_i);
	if ( !(noise.empty()) ){
		CV_Assert(res.rows == noise.rows);
		for ( int _i = 0; _i < res.rows; ++_i ){
			res.at<double>(_i) += noise.at<double>(_i);
		}
	}
	return res;
}
cv::Mat ISPKF_2D::ObsvGlobalQuaternion(const cv::Mat& inp, void* ptr, const cv::Mat noise)
{
	CV_Assert(inp.rows >= 9);
	CV_Assert(inp.cols == 1);
	CV_Assert(inp.type() == CV_64FC1);
	cv::Mat res(4, 1, CV_64FC1);
	for ( int _i = 0; _i < 4; ++_i ) res.at<double>(_i) = inp.at<double>(_i + 3);
	if ( !(noise.empty()) ){
		CV_Assert(res.rows == noise.rows);
		for ( int _i = 0; _i < res.rows; ++_i ){
			res.at<double>(_i) += noise.at<double>(_i);
		}
		Quaternion qtr(res.at<double>(0), res.at<double>(1), res.at<double>(2), res.at<double>(3));
		if ( qtr.getW() < 0. ) qtr.scalarmul(-1.);
		if ( qtr.isZero() ) qtr.setW(1.);
		qtr.normalize();
		res.at<double>(0) = qtr.getW();	res.at<double>(1) = qtr.getX();	res.at<double>(2) = qtr.getY();	res.at<double>(3) =qtr.getZ();		
	}	
	return res;
}
cv::Mat ISPKF_2D::ObsvLocalPlanarVels(const cv::Mat& inp, void* ptr, const cv::Mat noise)
{
	CV_Assert(inp.rows >= 9);
	CV_Assert(inp.cols == 1);
	CV_Assert(inp.type() == CV_64FC1);
	cv::Mat res(2, 1, CV_64FC1);
	for ( int _i = 0; _i < 2; ++_i ) res.at<double>(_i) = inp.at<double>(_i + 7);
	if ( !(noise.empty()) ){
		CV_Assert(res.rows == noise.rows);
		for ( int _i = 0; _i < res.rows; ++_i ){
			res.at<double>(_i) += noise.at<double>(_i);
		}
	}	
	return res;
}
cv::Mat ISPKF_2D::_correct_constraints(const cv::Mat& mat) const
{
	cv::Mat _res;	mat.copyTo(_res);
	Quaternion qtr(mat.at<double>(3), mat.at<double>(4), mat.at<double>(5), mat.at<double>(6));
	if ( qtr.getW() < 0. ) qtr.scalarmul(-1.);
	if ( qtr.isZero() ) qtr.setW(1.);
	qtr.normalize();
	_res.at<double>(3) = qtr.getW();	_res.at<double>(4) = qtr.getX();	_res.at<double>(5) = qtr.getY();	_res.at<double>(6) = qtr.getZ();
	return _res;
}

// The Base Class
ISPKF_MM::~ISPKF_MM()
{
	// nothing to delete here
}
ISPKF_MM::ISPKF_MM(const ISPKF_MM& other): SPKF(other)
{
	this->bTcam = other.bTcam;
	this->_initialized = other._initialized;
}
ISPKF_MM::ISPKF_MM(const unsigned int dimX, const double _alpha, const double _beta, const double _k): SPKF(dimX, _alpha, _beta, _k)
{
	// initialize the transform to standard transform b/w xyz robot to z-x-y
	CV_Assert(dimX >= 6);	// dimensions should always be greater than or equal to six : 3pos, 3angs. in xyz respectively
	this->_initialized = false;
	this->bTcam.setRotationFixed(cv::Matx31d(-M_PI_2,0,-M_PI_2));
}
TransformationMatrix ISPKF_MM::getBaseCameraTransform() const
{
	return this->bTcam;
}
void ISPKF_MM::setBaseCameraTransform(const TransformationMatrix& trmat)
{
	this->bTcam = trmat;
}
TransformationMatrix ISPKF_MM::getBasePose() const
{
	return TMatSS(this->getX());
}
TransformationMatrix ISPKF_MM::getCameraPose() const
{
	TransformationMatrix tinv = this->bTcam.getInverse();	// now it is camTbase
	TransformationMatrix res = tinv * this->getBasePose() * this->bTcam;
	return res;
}
void ISPKF_MM::setBasePose(const cv::Matx31d& globX, const cv::Matx31d& globAng)
{
	for ( int _i = 0; _i < 3; ++_i ){
		this->_X.at<double>(_i) = globX(_i);
	}
	Quaternion qtr = TransformationMatrix::XYZFixedToQuat(globAng);
	this->_X.at<double>(3) = qtr.getW();		this->_X.at<double>(4) = qtr.getX();
	this->_X.at<double>(5) = qtr.getY();		this->_X.at<double>(6) = qtr.getZ();
}
void ISPKF_MM::setBasePose(const TransformationMatrix& pose)
{
	cv::Matx31d globX = pose.getTranslation();
	for ( int _i = 0; _i < 3; ++_i ){
		this->_X.at<double>(_i) = globX(_i);
	}
	Quaternion qtr = pose.getQuaternion();
	this->_X.at<double>(3) = qtr.getW();		this->_X.at<double>(4) = qtr.getX();
	this->_X.at<double>(5) = qtr.getY();		this->_X.at<double>(6) = qtr.getZ();
}
void ISPKF_MM::setCameraPose(const TransformationMatrix& pose)
{
	TransformationMatrix bpose = this->bTcam * pose * this->bTcam.getInverse();
	this->setBasePose(bpose);
}
TransformationMatrix ISPKF_MM::TMatSS(const cv::Mat& inp)
{
	TransformationMatrix tmat;
	cv::Matx31d told;
	CV_Assert(inp.rows >= 7);
	told(0) = inp.at<double>(0);	told(1) = inp.at<double>(1);	told(2) = inp.at<double>(2);
	Quaternion _qtr(inp.at<double>(3),inp.at<double>(4),inp.at<double>(5),inp.at<double>(6));
	tmat.setTranslation(told);
	tmat.setQuaternion(_qtr);
	return tmat;
}
cv::Mat ISPKF_MM::SSTmat(const TransformationMatrix& tmat)
{
	cv::Matx31d told;
	Quaternion qtr;
	cv::Mat res(7,1,CV_64FC1);
	told = tmat.getTranslation();
	qtr = tmat.getQuaternion();
	res.at<double>(0) = told(0);		res.at<double>(1) = told(1);		res.at<double>(2) = told(2);
	res.at<double>(3) = qtr.getW();		res.at<double>(4) = qtr.getX();		res.at<double>(5) = qtr.getY();		res.at<double>(6) = qtr.getZ();
	return res;
}
cv::Mat ISPKF_MM::ComputeVels(const TransformationMatrix& before, const TransformationMatrix& after, const double _dt)
{
	// returns a 6x1 vector 3x1 linear velocity , 3x1 angular velocity
	cv::Matx31d _oldorig, _deltatrans, _newtrans;
	_oldorig = before.getTranslation();
	_newtrans = after.getTranslation();
	Quaternion oldquat = before.getQuaternion();
	Quaternion newquat = after.getQuaternion();
	_deltatrans = oldquat.inv().RotatePoint(_newtrans - _oldorig);
	Quaternion deltaquat = oldquat.inv() * newquat;
	if ( deltaquat.getW() < 0) deltaquat.scalarmul(-1.);
	cv::Matx31d _eul = TransformationMatrix::QuatToXYZFixed(deltaquat);
	//		compute vels
	_deltatrans = _deltatrans * ( 1. / _dt);
	_eul = _eul * ( 1. / _dt);
	cv::Mat _res(6, 1, CV_64FC1);
	for ( int _i = 0; _i < 3 ; ++_i ){
		_res.at<double>(_i) = _deltatrans(_i);
		_res.at<double>(_i + 3) = _eul(_i);	
	}
	return _res;
}
