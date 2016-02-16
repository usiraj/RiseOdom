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

#include "transformation_matrix.h"
#include <math.h>
#include <fstream>

TransformationMatrix::TransformationMatrix(){
	this->_initeye();
}
TransformationMatrix::TransformationMatrix(const cv::Matx34d &mat){
	this->trmatrix = mat;
}
TransformationMatrix::TransformationMatrix(const cv::Matx44d &mat){
	this->setMatrix(mat);
}
TransformationMatrix::TransformationMatrix(const cv::Mat &mat){
	for ( unsigned int _r = 0; (_r < 3) && (_r < mat.rows); ++_r){
		for ( unsigned int _c = 0; (_c < 4) && ( _c < mat.cols) ; ++_c){
			if ( mat.depth() == CV_64F ){
				this->trmatrix(_r, _c) = mat.at<double>(_r, _c);
			} else {
				this->trmatrix(_r, _c) = mat.at<float>(_r, _c);
			}
		}
	}
}
TransformationMatrix::TransformationMatrix(const cv::Matx31d &translation){
	this->_initeye();
	this->setTranslation(translation);
}
TransformationMatrix::TransformationMatrix(const cv::Matx33d &rot){
	this->setRotation(rot);
	this->trmatrix(0, 3) = 0; this->trmatrix(1, 3) = 0; this->trmatrix(2, 3) = 0;
}
TransformationMatrix::TransformationMatrix(const cv::Matx33d &rot, const cv::Matx31d &translation)
{
	this->setRotation(rot);
	this->setTranslation(translation);
}
TransformationMatrix::TransformationMatrix(const TransformationMatrix &cpy){
	this->trmatrix = cpy.trmatrix;
}
TransformationMatrix::~TransformationMatrix(){
}
// apply function
cv::Mat TransformationMatrix::applyTransform(const cv::Mat& _pts3d) const
{
	cv::Mat _res(_pts3d.rows, _pts3d.cols, _pts3d.type());
	if ( _pts3d.type() == CV_64FC1 && _pts3d.cols == 3 ){			// double single channel
		for ( int _i = 0; _i < _pts3d.rows; ++_i ){
			_res.at<double>(_i,0) = this->trmatrix(0,0)*_pts3d.at<double>(_i,0) + this->trmatrix(0,1)*_pts3d.at<double>(_i,1) + 
			this->trmatrix(0,2)*_pts3d.at<double>(_i,2) + this->trmatrix(0,3);
			_res.at<double>(_i,1) = this->trmatrix(1,0)*_pts3d.at<double>(_i,0) + this->trmatrix(1,1)*_pts3d.at<double>(_i,1) + 
			this->trmatrix(1,2)*_pts3d.at<double>(_i,2) + this->trmatrix(1,3);
			_res.at<double>(_i,2) = this->trmatrix(2,0)*_pts3d.at<double>(_i,0) + this->trmatrix(2,1)*_pts3d.at<double>(_i,1) + 
			this->trmatrix(2,2)*_pts3d.at<double>(_i,2) + this->trmatrix(2,3);
		}
	} else if ( _pts3d.type() == CV_64FC3 && _pts3d.cols == 1 ){	// double 3 channel
		for ( int _i = 0; _i < _pts3d.rows; ++_i ){
			cv::Point3d _pt = _pts3d.at<cv::Point3d>(_i);
			_res.at<cv::Point3d>(_i).x = this->trmatrix(0,0)*_pt.x + this->trmatrix(0,1)*_pt.y + this->trmatrix(0,2)*_pt.z + this->trmatrix(0,3);
			_res.at<cv::Point3d>(_i).y = this->trmatrix(1,0)*_pt.x + this->trmatrix(1,1)*_pt.y + this->trmatrix(1,2)*_pt.z + this->trmatrix(1,3);
			_res.at<cv::Point3d>(_i).z = this->trmatrix(2,0)*_pt.x + this->trmatrix(2,1)*_pt.y + this->trmatrix(2,2)*_pt.z + this->trmatrix(2,3);
		}
	} else 	if ( _pts3d.type() == CV_32FC1 && _pts3d.cols == 3 ){	// float single channel
		for ( int _i = 0; _i < _pts3d.rows; ++_i ){
			_res.at<float>(_i,0) = this->trmatrix(0,0)*_pts3d.at<float>(_i,0) + this->trmatrix(0,1)*_pts3d.at<float>(_i,1) + 
			this->trmatrix(0,2)*_pts3d.at<float>(_i,2) + this->trmatrix(0,3);
			_res.at<float>(_i,1) = this->trmatrix(1,0)*_pts3d.at<float>(_i,0) + this->trmatrix(1,1)*_pts3d.at<float>(_i,1) + 
			this->trmatrix(1,2)*_pts3d.at<float>(_i,2) + this->trmatrix(1,3);
			_res.at<float>(_i,2) = this->trmatrix(2,0)*_pts3d.at<float>(_i,0) + this->trmatrix(2,1)*_pts3d.at<float>(_i,1) + 
			this->trmatrix(2,2)*_pts3d.at<float>(_i,2) + this->trmatrix(2,3);
		}	
	} else if ( _pts3d.type() == CV_32FC3 && _pts3d.cols == 1 ){	// float 3 channel
		for ( int _i = 0; _i < _pts3d.rows; ++_i ){
			cv::Point3f _pt = _pts3d.at<cv::Point3f>(_i);
			_res.at<cv::Point3f>(_i).x = float(this->trmatrix(0,0)*_pt.x + this->trmatrix(0,1)*_pt.y + this->trmatrix(0,2)*_pt.z + this->trmatrix(0,3));
			_res.at<cv::Point3f>(_i).y = float(this->trmatrix(1,0)*_pt.x + this->trmatrix(1,1)*_pt.y + this->trmatrix(1,2)*_pt.z + this->trmatrix(1,3));
			_res.at<cv::Point3f>(_i).z = float(this->trmatrix(2,0)*_pt.x + this->trmatrix(2,1)*_pt.y + this->trmatrix(2,2)*_pt.z + this->trmatrix(2,3));
		}
	} else {
		CV_Error(CV_StsBadSize, "size of 3d points mat is incorrect");
	}
	return _res;
}
cv::Point3f TransformationMatrix::transformPoint(const cv::Point3f& _pt) const
{
	cv::Point3f _res;
	_res.x = float(this->trmatrix(0,0)*_pt.x + this->trmatrix(0,1)*_pt.y + this->trmatrix(0,2)*_pt.z + this->trmatrix(0,3));
	_res.y = float(this->trmatrix(1,0)*_pt.x + this->trmatrix(1,1)*_pt.y + this->trmatrix(1,2)*_pt.z + this->trmatrix(1,3));
	_res.z = float(this->trmatrix(2,0)*_pt.x + this->trmatrix(2,1)*_pt.y + this->trmatrix(2,2)*_pt.z + this->trmatrix(2,3));
	return _res;
}
// get members
void TransformationMatrix::inverse()
{
	cv::Matx33d _rmat;
	cv::Matx31d _tvec = this->getTranslation();
	cv::transpose(this->getRotation(), _rmat);
	_tvec = -(_rmat * _tvec);
	this->setRotation(_rmat);
	this->setTranslation(_tvec);
}
cv::Matx44d TransformationMatrix::getInverse() const{
	TransformationMatrix mat(*this);
	mat.inverse();
	return mat.getMatrix();	
}
cv::Matx44d TransformationMatrix::getMatrix() const{
	cv::Matx44d _res;
	for (unsigned int _r = 0; _r < 3; ++_r){
		for ( unsigned int _c = 0; _c < 4; ++_c){
			_res(_r, _c) = this->trmatrix(_r, _c);
		}
	}
	_res(3, 0) = 0; _res(3, 1) = 0; _res(3, 2) = 0; _res(3, 3) = 1;
	return _res;
}
cv::Matx31d TransformationMatrix::getTranslation() const{
	cv::Matx31d _tr;
	for (unsigned int _r = 0; _r < 3; ++_r){
		_tr(_r, 0) = this->trmatrix(_r, 3);
	}
	return _tr;
}
cv::Matx33d TransformationMatrix::getRotation() const{
	cv::Matx33d _rot;
	for ( unsigned int _r = 0; _r < 3; ++_r){
		for (unsigned int _c = 0; _c  < 3; ++_c){
			_rot(_r, _c) = this->trmatrix(_r, _c);
		}
	}
	return _rot;
}
cv::Matx31d TransformationMatrix::getRotationFixed() const{
	return TransformationMatrix::RotToFixedXYZ(this->getRotation());
}
cv::Mat TransformationMatrix::getRodrigues() const
{
	cv::Mat _res;
	cv::Rodrigues(cv::Mat(this->getRotation()), _res);
	return _res;
}
Quaternion TransformationMatrix::getQuaternion() const{
	Quaternion res = TransformationMatrix::RotToQuat(this->getRotation());
	return res;
}
// set members
void TransformationMatrix::setMatrix(const cv::Matx44d &mat){
	for (unsigned int _r = 0; _r < 3; ++_r){
		for ( unsigned int _c = 0; _c < 4; ++_c){
			this->trmatrix(_r, _c) = mat(_r, _c);
		}
	}
}
void TransformationMatrix::setMatrix(const cv::Matx34d &mat){
	this->trmatrix = mat;
}
void TransformationMatrix::setTranslation(const cv::Matx31d &translation){
	for ( unsigned int _r = 0; _r < 3; ++_r){
		this->trmatrix(_r, 3) = translation(_r, 0);
	}
}
void TransformationMatrix::setTranslation(const cv::Mat& translation)
{
	for ( unsigned int _r = 0; _r < 3; ++_r){
		if ( translation.depth() == CV_64F ){
			this->trmatrix(_r, 3) = translation.at<double>(_r);
		} else {
			this->trmatrix(_r, 3) = translation.at<float>(_r);
		}
	}
}
void TransformationMatrix::setRotation(const cv::Mat& rotation)
{
	for ( unsigned int _r = 0; _r < 3; ++_r){
		for (unsigned int _c = 0; _c  < 3; ++_c){
			if ( rotation.depth() == CV_64F ){
				this->trmatrix(_r, _c) = rotation.at<double>(_r, _c);
			} else {
				this->trmatrix(_r, _c) = rotation.at<float>(_r, _c);
			}
		}
	}
}
void TransformationMatrix::setRodrigues(const cv::Mat& _rvec, const cv::Mat& _tvec)
{
	cv::Matx33d _rmat;
	cv::Rodrigues(_rvec, _rmat);
	this->setRotation(_rmat);
	this->setTranslation(_tvec);
}
void TransformationMatrix::setRotation(const cv::Matx33d &rotation){
	for ( unsigned int _r = 0; _r < 3; ++_r){
		for (unsigned int _c = 0; _c  < 3; ++_c){
			this->trmatrix(_r, _c) = rotation(_r, _c);
		}
	}
}
void TransformationMatrix::setRotationFixed(const cv::Matx31d &rot){
	cv::Matx33d rmat = TransformationMatrix::XYZFixedToRot(rot);
	this->setRotation(rmat);
}
void TransformationMatrix::setQuaternion(const Quaternion &qtr){
	cv::Matx33d _rmat = TransformationMatrix::QuatToRot(qtr);
	this->setRotation(_rmat);
}
TransformationMatrix& TransformationMatrix::operator=(const TransformationMatrix &rhs){
	this->setMatrix(rhs.getMatrix());
	return *this;
}
TransformationMatrix& TransformationMatrix::operator=(const cv::Matx44d &rhs){
	this->setMatrix(rhs);
	return *this;
}
TransformationMatrix& TransformationMatrix::operator=(const cv::Matx34d &rhs){
	this->setMatrix(rhs);
	return *this;
}
TransformationMatrix TransformationMatrix::operator*(const TransformationMatrix& rhs) const
{
	TransformationMatrix _res((this->getMatrix() * rhs.getMatrix()));
	return _res;
}
TransformationMatrix TransformationMatrix::operator*(const cv::Matx44d& rhs) const
{
	TransformationMatrix _res((this->getMatrix() * rhs));
	return _res;
}
void TransformationMatrix::print() const
{
	for ( int _i = 0; _i < 3; ++_i){
		for (int _j = 0; _j < 4; ++_j){
			printf("%lf\t", this->trmatrix(_i, _j));
		}
		printf("\n");
	}
	printf("%lf\t%lf\t%lf\t%lf\n", 0., 0., 0., 1.);
}
//helpers
void TransformationMatrix::_initeye()
{
	this->trmatrix.all(0);
	this->trmatrix(0, 0) = 1;
	this->trmatrix(1, 1) = 1;
	this->trmatrix(2, 2) = 1;
}
cv::Matx33d TransformationMatrix::XYZFixedToRot(const cv::Matx31d &xyz){
	double _c3 = cos(xyz(0, 0));
	double _c2 = cos(xyz(1, 0));
	double _c1 = cos(xyz(2, 0));
	double _s1 = sin(xyz(2, 0));
	double _s2 = sin(xyz(1, 0));
	double _s3 = sin(xyz(0, 0));
    double r11 = _c1 * _c2;
    double r12 = (_c1 * _s2 * _s3) - (_s1 * _c3);
    double r13 = (_c1 * _s2 * _c3) + (_s1 * _s3);
    double r21 = _s1 * _c2;
    double r22 = (_s1 * _s2 * _s3) + (_c1 * _c3);
    double r23 = (_s1 * _s2 * _c3) - (_c1 * _s3);
    double r31 = -_s2;
    double r32 = _c2 * _s3;
    double r33 = _c2 * _c3;
	cv::Matx33d _rmat;
	_rmat << r11, r12, r13, r21, r22, r23, r31, r32, r33;
    return _rmat;
}
cv::Matx31d TransformationMatrix::RotToFixedXYZ(const cv::Matx33d &rot){
	double r00 = rot(0, 0);
	double r10 = rot(1, 0);
	double _a = sqrt((r00*r00)+(r10*r10));
	double _th = 1e-16;
	double _beta1 = atan2(-rot(2, 0), _a);
	double _sbeta1 = sin(_beta1);
	double _x, _y , _z;
	if ((1. - (_sbeta1 * _sbeta1)) < sqrt(_th)){
		if ( _sbeta1 > 0. ){
			_y = M_PI_2 ;
			_z = 0;
			_x = atan2(rot(0, 1), rot(1, 1));
		} else {
			_y = - M_PI_2;
			_z = 0;
			_x =  -atan2(rot(0, 1), rot(1, 1));
		}	
	} else {
		_y = _beta1;
		double _cb1 = cos(_beta1);
		_z = atan2(rot(1, 0)/_cb1, r00/_cb1);
		_x = atan2(rot(2, 1)/_cb1, rot(2, 2)/_cb1);
	}
	cv::Matx31d _res;
	_res << _x , _y, _z;
	return _res;
}
cv::Matx33d TransformationMatrix::QuatToRot(const Quaternion& quat)
{
	double r11 = 1. - 2. * ( (quat.getY() * quat.getY()) + (quat.getZ() * quat.getZ()) );
    double r12 = 2. * ( (quat.getX() * quat.getY()) - (quat.getZ()*quat.getW()) ) ;
    double r13 = 2. * ( (quat.getX()*quat.getZ()) + (quat.getY()*quat.getW()) );
    double r21 = 2. * ( (quat.getX()*quat.getY()) + (quat.getZ()*quat.getW()) );
    double r22 = 1. - 2. * ( (quat.getX() * quat.getX()) + (quat.getZ()*quat.getZ()) );
    double r23 = 2. * ( (quat.getY()*quat.getZ()) - (quat.getX()*quat.getW()) );
    double r31 = 2. * ( (quat.getX()*quat.getZ()) - (quat.getY()*quat.getW()) );
    double r32 = 2. * ( (quat.getY()*quat.getZ()) + (quat.getX()*quat.getW()) );
    double r33 = 1. - 2. * ( (quat.getX()*quat.getX()) + (quat.getY()*quat.getY()) );
	cv::Matx33d _rmat;
	_rmat << r11, r12, r13, r21, r22, r23, r31, r32, r33;
    return _rmat;
}
Quaternion TransformationMatrix::RotToQuat(const cv::Matx33d& rot)
{
	// conversion to quaternion
	double tr = rot(0,0) + rot(1,1) + rot(2,2);
	double qw, qx, qy, qz;
	if (tr > 0) { 
		double S = sqrt(tr + 1.0) * 2.; // S=4*qw   
		qw = 0.25 * S;
		qx = (rot(2,1) - rot(1,2)) / S;
		qy = (rot(0,2) - rot(2,0)) / S; 
		qz = (rot(1,0) - rot(0,1)) / S; 
	} else if ((rot(0,0) > rot(1,1)) && (rot(0,0) > rot(2,2))) { 
		double S = sqrt(1.0 + rot(0,0) - rot(1,1) - rot(2,2)) * 2.; // S=4*qx 
		qw = (rot(2,1) - rot(1,2)) / S;
		qx = 0.25 * S;
		qy = (rot(0,1) + rot(1,0)) / S; 
		qz = (rot(0,2) + rot(2,0)) / S; 
	} else if (rot(1,1) > rot(2,2)) { 
		double S = sqrt(1.0 + rot(1,1) - rot(0,0) - rot(2,2)) * 2.; // S=4*qy
		qw = (rot(0,2) - rot(2,0)) / S;
		qx = (rot(0,1) + rot(1,0)) / S; 
		qy = 0.25 * S;
		qz = (rot(1,2) + rot(2,1)) / S; 
	} else { 
		double S = sqrt(1.0 + rot(2,2) - rot(0,0) - rot(1,1)) * 2.; // S=4*qz
		qw = (rot(1,0) - rot(0,1)) / S;
		qx = (rot(0,2) + rot(2,0)) / S;
		qy = (rot(1,2) + rot(2,1)) / S;
		qz = 0.25 * S;
}	
	Quaternion res(qw,qx,qy,qz);
	// fix for -q
	if ( res.getW() < 0. ) res.scalarmul(-1.);
	return res;
}
Quaternion TransformationMatrix::XYZFixedToQuat(const cv::Matx31d& xyz)
{
	double cz = cos(xyz(2) / 2.);
	double sz = sin(xyz(2) / 2.);
	double cy = cos(xyz(1) / 2.);
	double sy = sin(xyz(1) / 2.);
	double cx = cos(xyz(0) / 2.);
	double sx = sin(xyz(0) / 2.);
	double _w, _x, _y, _z;
	_w = (cx * cy * cz ) + ( sx * sy * sz );
	_x = (sx * cy *  cz ) - ( cx * sy * sz );
	_y = ( cx * sy * cz ) + ( sx * cy * sz );
	_z = ( cx * cy * sz ) - ( sx * sy * cz );
	Quaternion res(_w, _x, _y, _z);
	// fix for -q
	if ( res.getW() < 0. ) res.scalarmul(-1.);
	return res;
}
cv::Matx31d TransformationMatrix::QuatToXYZFixed(const Quaternion& quat)
{
	double _th = 0.5 - 1e-10;
	double _qw, _qx, _qy, _qz;
	double _den3, _num3, _den1, _num1;
	_qw = quat.getW();	_qx = quat.getX();	_qy = quat.getY();	_qz = quat.getZ();
	double test = (_qw * _qy) - (_qx * _qz);
	double _x, _y, _z;
	cv::Matx31d _res;
	if ( test > _th ){
		_y = M_PI_2;
		_z = 0.;
		_x = 2. * atan2(_qx, _qw);
	} else if ( test < -_th ){
		_y = -M_PI_2;
		_z = 0.;
		_x = 2. * atan2(_qx, _qw);
	} else {
		_y = asin(2. * test);		
		_num3 = 2.* ( (_qw*_qz) + (_qx * _qy) );
		_num1 = 2.* ( (_qw*_qx) + (_qy * _qz) );
		_den3 = 1. - 2. * ( (_qy * _qy) + (_qz * _qz) );
		_den1 = 1. - 2. * ( (_qx * _qx) + (_qy * _qy) );
		_z = atan2(_num3 , _den3 );
		_x = atan2(_num1 , _den1 );		
	}
	_res << _x, _y, _z;
	return _res;
}
// utility functions
double angdiff_minimum(const double ang1, const double ang2)
{
	double ang = angdiff_singed(ang1, ang2);
	if ( ang < 0) ang = -ang;
	return ang;
}
double angdiff_singed(const double angnew, const double angold)
{
	double diff = angnew - angold;
	while ( diff < -M_PI_2 ) diff += M_PI;
	while ( diff > M_PI_2 ) diff -= M_PI;
	return diff;
}

void save_poses(const char* filename, const std::vector< TransformationMatrix >& _translist)
{
	std::ofstream _file(filename);
	if ( !_file ) fprintf(stderr, "Cannot write pose to file : %s\n", filename);
	else {
		_file << std::scientific ;
		for ( std::vector<TransformationMatrix>::const_iterator itr = _translist.begin(); itr < _translist.end(); ++itr ){
			cv::Matx44d _tr = itr->getMatrix();
			for ( unsigned int _i = 0; _i < 12; ++_i ){
				unsigned int _c = _i % 4;
				unsigned int _r = _i / 4;
				_file << _tr(_r, _c) << " ";
			}
			_file << std::endl;
		}
	}
	_file.close();
}

void save_mit(const char* filename, const std::vector< int >& mtchs, const std::vector< int >& inliers,
			  const std::vector< double >& timings)
{
	std::ofstream _file(filename);
	if ( !_file ) fprintf(stderr, "Cannot write pose to file : %s\n", filename);
	else {
		for ( unsigned int _i = 0; _i < mtchs.size(); ++_i){
			_file << mtchs[_i]	<< " " << inliers[_i] << " " << timings[_i] << std::endl;
		}
	}
	_file.close();
}
// Quaternions
Quaternion::Quaternion()
{
	this->w = 1.;
	this->x = 0.;
	this->y = 0.;
	this->z = 0.;
}
Quaternion::Quaternion(const double& _w, const double& _x, const double& _y, const double& _z)
{
	this->x = _x;
	this->y = _y;
	this->z = _z;
	this->w = _w;
}
Quaternion::Quaternion(const Quaternion& copy)
{
	this->w = copy.w;
	this->x = copy.x;
	this->y = copy.y;
	this->z = copy.z;
}
Quaternion::~Quaternion()
{ }
double Quaternion::getW() const
{
	return this->w;
}
double Quaternion::getX() const
{
	return this->x;
}
double Quaternion::getY() const
{
	return this->y;
}
double Quaternion::getZ() const
{
	return this->z;
}
void Quaternion::setW(const double& _w)
{
	this->w = _w;
}
void Quaternion::setX(const double& _x)
{
	this->x = _x;
}
void Quaternion::setY(const double& _y)
{
	this->y = _y;
}
void Quaternion::setZ(const double& _z)
{
	this->z = _z;
}
void Quaternion::set(const Quaternion& quat)
{
	this->w = quat.w;
	this->x = quat.x;
	this->y = quat.y;
	this->z = quat.z;
}
void Quaternion::set(const double& _w, const double& _x, const double& _y, const double& _z)
{
	this->w = _w;
	this->x = _x;
	this->y = _y;
	this->z = _z;
}
void Quaternion::add(const Quaternion& quat)
{
	this->w += quat.w;
	this->x += quat.x;
	this->y += quat.y;
	this->z += quat.z;
}
void Quaternion::sub(const Quaternion& quat)
{
	this->w -= quat.w;
	this->x -= quat.x;
	this->y -= quat.y;
	this->z -= quat.z;
}
void Quaternion::scalarmul(const double& num)
{
	this->w *= num;
	this->x *= num;
	this->y *= num;
	this->z *= num;
}
void Quaternion::mul(const Quaternion& quat)
{
	double _w, _x, _y, _z;
	_w = (this->w * quat.w) - (this->x * quat.x) - (this->y * quat.y) - (this->z * quat.z) ;
	_x = (this->x * quat.w ) + (this->w * quat.x ) + (this->y * quat.z ) - ( this->z * quat.y );
	_y = (this->w * quat.y) - ( this->x * quat.z ) + (this->y * quat.w ) + (this->z * quat.x );
	_z = (this->w * quat.z ) + (this->x * quat.y ) - (this->y * quat.x ) + (this->z * quat.w );
	this->w = _w;	this->x = _x;
	this->y = _y;	this->z = _z;
}
void Quaternion::conjugate()
{
	this->x = -(this->x);
	this->y = -(this->y);
	this->z = -(this->z);
}
Quaternion Quaternion::conj() const
{
	Quaternion res(*this);
	res.conjugate();
	return res;
}
void Quaternion::inverse()
{
	this->normalize();
	this->conjugate();
}
Quaternion Quaternion::inv() const
{
	Quaternion res(*this);
	res.inverse();
	return res;
}
double Quaternion::norm2() const
{
	double _sqr = 0.;
	_sqr += this->w * this->w;
	_sqr += this->x * this->x;
	_sqr += this->y * this->y;
	_sqr += this->z * this->z;
	return _sqr;
}
double Quaternion::dist(const Quaternion& q2) const
{
	Quaternion diff = (*this) - q2;
	return diff.norm();
}
double Quaternion::distrot(const Quaternion& q2) const
{
	// check w is +ive for both
	Quaternion _q1, _q2;
	_q1 = *this;
	_q2 = q2;
	if ( _q1.getW() < 0. ) _q1.scalarmul(-1.);
	if ( _q2.getW() < 0. ) _q2.scalarmul(-1.);
	if ( !_q1.isUnit() ) _q1.normalize();
	if ( !_q2.isUnit() ) _q2.normalize();
	// normalize both
	return _q1.dist(_q2);
}
double Quaternion::norm() const
{
	double _sqr = this->norm2();
	return sqrt(_sqr);
}
void Quaternion::normalize()
{
	double _nrm = this->norm();
	this->w /= _nrm;
	this->x /= _nrm;
	this->y /= _nrm;
	this->z /= _nrm;
}
Quaternion Quaternion::normalized() const
{
	Quaternion res(*this);
	res.normalize();
	return res;
}
Quaternion& Quaternion::operator=(const Quaternion& rhs)
{
	this->set(rhs);
	return (*this);
}
bool Quaternion::operator==(const Quaternion& cmp) const
{
	return !((*this) != cmp);
}
bool Quaternion::operator!=(const Quaternion& cmp) const
{
	if ( this->w != cmp.w ) return false;
	if ( this->x != cmp.x ) return false;
	if ( this->y != cmp.y ) return false;
	if ( this->z != cmp.z ) return false;
	return true;
}
Quaternion& Quaternion::operator+=(const Quaternion& rhs)
{
	this->add(rhs);
	return *this;
}
Quaternion Quaternion::operator+(const Quaternion& rhs) const
{
	Quaternion ans(*this);
	ans.add(rhs);
	return ans;
}
Quaternion& Quaternion::operator-=(const Quaternion& rhs)
{
	this->sub(rhs);
	return *this;
}
Quaternion Quaternion::operator-(const Quaternion& rhs) const
{
	Quaternion ans(*this);
	ans.sub(rhs);
	return ans;
}
Quaternion& Quaternion::operator*=(const Quaternion& rhs)
{
	this->mul(rhs);
	return *this;
}

Quaternion Quaternion::operator*(const Quaternion& rhs) const
{
	Quaternion ans(*this);
	ans.mul(rhs);
	return ans;
}
void Quaternion::print() const
{
	printf("( %lf, i %lf, j %lf, k %lf )",this->w, this->x, this->y, this->z);
}
void Quaternion::prints() const
{
	printf("( %.10le, i %.10le, j %.10le, k %.10le )",this->w, this->x, this->y, this->z);
}
bool Quaternion::isZero() const
{
	double _nrm = this->norm2();
	if ( _nrm < 1E-16 ) return true;
	return false;
}
bool Quaternion::isUnit() const
{
	double _nrm = this->norm2() - 1.; _nrm  = ( _nrm < 0. )? -1.*_nrm : _nrm;
	if ( _nrm < 1E-16 ) return true;
	return false;
}
cv::Matx31d Quaternion::RotatePoint(const cv::Matx31d& pt) const
{
	Quaternion _pt(0., pt(0), pt(1), pt(2) );
	Quaternion ans = (*this) * _pt * this->conj();
	cv::Matx31d res;
	res << ans.getX(), ans.getY(), ans.getZ();
	return res;
}
cv::Mat Quaternion::RotatePoints(const cv::Mat& pt) const
{
	CV_Assert( pt.rows == 3);
	CV_Assert( pt.cols > 0);
	CV_Assert( pt.type() == CV_64FC1 );
	Quaternion _temp, _pt, _conj;
	_conj = this->conj();
	cv::Mat res(3, pt.cols, CV_64FC1 );
	for ( int _i = 0; _i < pt.cols; ++_i ){
		_pt.set(0., pt.at<double>(0,_i), pt.at<double>(1,_i), pt.at<double>(2,_i));
		_temp = (*this) * _pt * _conj;
		res.at<double>(0, _i) = _temp.getX();
		res.at<double>(1, _i) = _temp.getY();
		res.at<double>(2, _i) = _temp.getZ();
	}
	return res;
}























