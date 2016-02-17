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

#ifndef TRANSFORMATIONMATRIX_H
#define TRANSFORMATIONMATRIX_H

#define MAXIMUM_POSSIBLE_VEL 50.

#include <opencv2/opencv.hpp>
#include <vector>

class Quaternion {
public:
	Quaternion();	// default constructor
	Quaternion(const double &_w, const double &_x, const double &_y, const double &_z);
	Quaternion(const Quaternion &copy);
	virtual ~Quaternion();
	// some basic members
	void add(const Quaternion &quat);
	void sub(const Quaternion &quat);
	void mul(const Quaternion &quat);
	void scalarmul(const double &num);
	virtual void inverse();
	virtual Quaternion inv() const;
	void conjugate();
	Quaternion conj() const;
	void normalize();
	Quaternion normalized() const;
	double norm() const;
	double dist(const Quaternion &q2) const;
	double distrot(const Quaternion &q2) const;		// distance for +ive w and unit quaternion
	double norm2() const;
	double getW() const;
	double getX() const;
	double getY() const;
	double getZ() const;
	void setW(const double &_w);
	void setX(const double &_x);
	void setY(const double &_y);
	void setZ(const double &_z);
	virtual void set(const Quaternion &quat);
	virtual void set(const double &_w, const double &_x, const double &_y, const double &_z);
	cv::Matx31d RotatePoint(const cv::Matx31d &pt) const;
 	cv::Mat RotatePoints(const cv::Mat &pt) const;
	// some operators
	bool isUnit() const;
	bool isZero() const;
	bool operator==(const Quaternion &cmp) const;
	bool operator!=(const Quaternion &cmp) const;
	Quaternion& operator=(const Quaternion &rhs);
	Quaternion& operator+=(const Quaternion &rhs);
	Quaternion& operator-=(const Quaternion &rhs);
	Quaternion& operator*=(const Quaternion &rhs);
	Quaternion operator+(const Quaternion &rhs) const;
	Quaternion operator-(const Quaternion &rhs) const;
	Quaternion operator*(const Quaternion &rhs) const;
	void print() const;
	void prints() const;
protected:
	double w;
	double x;
	double y;
	double z;
};

class TransformationMatrix
{		// change so that internal representation is by quaternions
public:
	TransformationMatrix();
	TransformationMatrix(const cv::Matx34d &mat);
	TransformationMatrix(const cv::Matx44d &mat);
	TransformationMatrix(const cv::Mat &mat);
	TransformationMatrix(const cv::Matx31d &translation);
	TransformationMatrix(const cv::Matx33d &rot);
	TransformationMatrix(const cv::Matx33d &rot, const cv::Matx31d &translation);
	TransformationMatrix(const TransformationMatrix &cpy);
	virtual ~TransformationMatrix();
	void inverse();
	cv::Mat applyTransform(const cv::Mat &_pts3d) const;
	cv::Point3f transformPoint(const cv::Point3f &pt) const;
	// get members
	cv::Matx44d getInverse() const;
	cv::Matx44d getMatrix() const;
	cv::Matx31d getTranslation() const;
	cv::Matx33d getRotation() const;
	cv::Mat getRodrigues() const;
	cv::Matx31d getRotationFixed() const;
	Quaternion getQuaternion() const;
	// set members
	void setMatrix(const cv::Matx44d &mat);
	void setMatrix(const cv::Matx34d &mat);
	void setTranslation(const cv::Matx31d &translation);
	void setTranslation(const cv::Mat &translation);
	void setRotation(const cv::Mat &rotation);
	void setRotation(const cv::Matx33d &rotation);
	void setRotationFixed(const cv::Matx31d &rot);
	void setQuaternion(const Quaternion &qtr);
	void setRodrigues(const cv::Mat &_rvec,const cv::Mat &_tvec);
	TransformationMatrix& operator=(const TransformationMatrix &rhs);
	TransformationMatrix& operator=(const cv::Matx44d &rhs);
	TransformationMatrix& operator=(const cv::Matx34d &rhs);
	TransformationMatrix operator*(const cv::Matx44d &rhs) const;
	TransformationMatrix operator*(const TransformationMatrix &rhs) const;
	void print() const;	
	// static functions
	static cv::Matx33d XYZFixedToRot(const cv::Matx31d &xyz);	// or zyx euler angles : for which vector is x y z
	static cv::Matx31d RotToFixedXYZ(const cv::Matx33d &rot);
	static cv::Matx33d QuatToRot(const Quaternion &quat);
	static Quaternion RotToQuat(const cv::Matx33d &rot);
	static Quaternion XYZFixedToQuat(const cv::Matx31d &xyz);	// or zyx euler angles : for which vector is x y z
	static cv::Matx31d QuatToXYZFixed( const Quaternion &quat); 
protected:
	cv::Matx34d trmatrix;
private:
	void _initeye();
};

void save_poses(const char *filename, const std::vector<TransformationMatrix> &_translist);
void save_mit(const char *filename, const std::vector<int> &mtchs, const std::vector<int> &inliers, const std::vector<double> &timings);

double angdiff_singed(const double angnew, const double angold);
double angdiff_minimum(const double ang1, const double ang2);

#endif // TRANSFORMATIONMATRIX_H
