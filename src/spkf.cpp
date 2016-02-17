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

#include "spkf.h"
#include <stdio.h>
#include "utils.h"
// fitler public functions
void SPKF::SigmaPrediction(const double delt, void* ptr)
{
	this->_sigma_prediction(delt, ptr);
}
void SPKF::MeasurementUpdate(const cv::Mat& z, const int idx, void* ptrdata, bool iterative)
{
	CV_Assert(idx < this->_obsrvfunc.size());
	CV_Assert(this->_obsrvfunc.at(idx) != NULL);
	try{
		if ( iterative ) this->_update_measurement_ispkf(z, idx, ptrdata);
		else this->_update_measurement_ukf(z, idx, ptrdata);
	} catch ( cv::Exception &e){
		if ( e.code != CV_StsNoConv ) throw(e);
	}
}
void SPKF::StartFilter()
{
	// A simple base start
	this->Start();
}
void SPKF::Start()
{
	// initialize and augment states, calculate lambda , weights , etc  and do other necessary steps
	this->_X = this->_correct_constraints(this->_X);
	this->_P = cv::Mat::eye(this->Lx, this->Lx, CV_64FC1) * 1E-3;
	if ( this->_statetransition == NULL ) CV_Error( CV_StsError,"State transiton not set");
}
void SPKF::_sigma_prediction(const double delt, void* ptr)
{
	// calculate weights and sigma points
	cv::Mat wtsm, wtsc, sigmat, _augP, _augX;
	int Laug = 2 * this->Lx;
	double lambda = SPKF::calculate_lambda(Laug, this->alpha, this->k);
	// augment states
	_augX = cv::Mat::zeros(Laug, 1, CV_64FC1);		this->_X.copyTo(_augX(cv::Rect(0, 0, 1, this->Lx)));
	_augP = cv::Mat::zeros(Laug, Laug, CV_64FC1);	this->_P.copyTo(_augP(cv::Rect(0,0,this->Lx,this->Lx)));
	this->_Q.copyTo(_augP(cv::Rect(this->Lx,this->Lx,this->Lx,this->Lx)));
	// caculate matix for sigma points
	sigmat = SPKF::calculate_sigmamat(_augP, lambda, Laug);
	// calcuate weights
	wtsm = SPKF::calculate_weights_mean(Laug, lambda);
	wtsc = SPKF::calculate_weights_covariance(Laug, lambda, this->alpha, this->beta);
	// find new sigma points
	cv::Mat _sigpts = this->_batchcorrect_constraints(SPKF::calculate_sigmapts(_augX, sigmat));
	cv::Mat __sigpts, _noise;
	__sigpts = _sigpts(cv::Rect(0, 0, _sigpts.cols, this->Lx));
	_noise = _sigpts(cv::Rect(0,  this->Lx, _sigpts.cols, this->Lx));
	cv::Mat _xhat = this->_cvectorapply_predictionfunc(__sigpts , delt, ptr, _noise);
	// calculate mean and covariance
	this->_X = this->_correct_constraints(SPKF::calculate_Zmean( _xhat, wtsm ));
	this->_P = SPKF::calculate_covZ(_xhat, this->_X, wtsc) + this->_Q;
	this->_protectP();
}
void SPKF::_update_measurement_ispkf(const cv::Mat& z, const int idx, void* ptr)
{
	// calculate weights and sigma points
	cv::Mat wtsm, wtsc, sigmat, _augP, _augX, Rmat;
	Rmat = this->_R.at(idx);
	int Laug = this->Lx + Rmat.rows;
	double lambda = SPKF::calculate_lambda(Laug, this->alpha, this->k);
	// augmnet states
	_augX = cv::Mat::zeros(Laug, 1, CV_64FC1);		this->_X.copyTo(_augX(cv::Rect(0, 0, 1, this->Lx)));
	_augP = cv::Mat::zeros(Laug, Laug, CV_64FC1);	this->_P.copyTo(_augP(cv::Rect(0,0,this->Lx,this->Lx)));
	Rmat.copyTo(_augP(cv::Rect(this->Lx,this->Lx, Rmat.cols, Rmat.rows)));
	// caculate matix for sigma points
	sigmat = SPKF::calculate_sigmamat(_augP, lambda, Laug);
	// calcuate weights
	wtsm = SPKF::calculate_weights_mean(Laug, lambda);
	wtsc = SPKF::calculate_weights_covariance(Laug, lambda, this->alpha, this->beta);
	double _looperror = 10.0;
	bool _noconvergence = false;
	cv::Mat _sigpts, _noise, __sigpts, _zhat, _Zmean, _covZ, _covXZ;
	cv::Mat _noisesum, _invnsum, _invP, _kalmangain , _jacobianobservation , _obsresidual, _xi0, _xi1, _diff;
	// calcualte inv P
	cv::invert(this->_P, _invP, cv::DECOMP_CHOLESKY);
	_xi0 = this->getX();
	int _i;
	for ( _i = 0; _looperror > this->_convergence; ++_i ){
		// update jacobian and kalman gain
		_xi0.copyTo(_augX(cv::Rect(0,0, 1, this->Lx)));
		_sigpts = this->_batchcorrect_constraints(SPKF::calculate_sigmapts(_augX, sigmat));			__sigpts = _sigpts(cv::Rect(0, 0, _sigpts.cols, this->Lx ));
		_noise = _sigpts(cv::Rect(0,  this->Lx, _sigpts.cols, Rmat.rows));
		_zhat = this->_cvectorapply_observationfunc(_sigpts, idx, ptr, _noise);
		_Zmean = SPKF::calculate_Zmean(_zhat, wtsm);
		_covZ = SPKF::calculate_covZ(_zhat, _Zmean, wtsc) + Rmat;
		_covXZ = SPKF::calculate_covXZ(_zhat, _Zmean, __sigpts, _xi0, wtsc);
		_noisesum = _covZ + Rmat;
		cv::invert(_noisesum, _invnsum, cv::DECOMP_CHOLESKY);
		_jacobianobservation = _covXZ.t() * _invP ;
		_kalmangain = _covXZ * _invnsum;
		_obsresidual = z - _Zmean - (_jacobianobservation * ( this->_X - _xi0) );
		_xi1 =  this->_correct_constraints( this->_X + ( _kalmangain * _obsresidual) );
		_diff = _xi1 - _xi0; 	_looperror = cv::norm(_diff, cv::NORM_L2);
		_xi0 = _xi1.clone();
		// if convergence failed after maximum iterations
		if ( _i >= this->_max_iter ){
			_noconvergence = true;
			break;
		}
	}
	_xi0.copyTo(this->_X);
	// update P
	this->_P = this->_P - ( _kalmangain * _covXZ.t());
	this->_protectP();
	// if failed to converge throw exception
	if ( _noconvergence ){
		CV_Error(CV_StsNoConv , "ISPKF : iterative update failed to converge");
	}
}
void SPKF::_update_measurement_ukf(const cv::Mat& z, const int idx, void* ptr)
{
	// calculate weights and sigma points
	cv::Mat wtsm, wtsc, sigmat, _augP, _augX, Rmat;
	Rmat = this->_R.at(idx);
	int Laug = this->Lx + Rmat.rows;
	double lambda = SPKF::calculate_lambda(Laug, this->alpha, this->k);
	// augmnet states
	_augX = cv::Mat::zeros(Laug, 1, CV_64FC1);		this->_X.copyTo(_augX(cv::Rect(0, 0, 1, this->Lx)));
	_augP = cv::Mat::zeros(Laug, Laug, CV_64FC1);	this->_P.copyTo(_augP(cv::Rect(0,0,this->Lx,this->Lx)));
	Rmat.copyTo(_augP(cv::Rect(this->Lx,this->Lx, Rmat.cols, Rmat.rows)));
	// caculate matix for sigma points
	sigmat = SPKF::calculate_sigmamat(_augP, lambda, Laug);
	// calcuate weights
	wtsm = SPKF::calculate_weights_mean(Laug, lambda);
	wtsc = SPKF::calculate_weights_covariance(Laug, lambda, this->alpha, this->beta);
	// find new sigma points
	cv::Mat _sigpts = this->_batchcorrect_constraints(SPKF::calculate_sigmapts(_augX, sigmat));
	cv::Mat __sigpts, _noise;	
	__sigpts = _sigpts(cv::Rect(0, 0, _sigpts.cols, this->Lx));
	_noise = _sigpts(cv::Rect(0,  this->Lx, _sigpts.cols, Rmat.rows));
	cv::Mat _zhat = this->_cvectorapply_observationfunc(_sigpts, idx, ptr, _noise);
	cv::Mat _zmean = SPKF::calculate_Zmean(_zhat, wtsm);
	cv::Mat _zcov = SPKF::calculate_covZ(_zhat, _zmean, wtsc) + Rmat;
	cv::Mat _xzcov = SPKF::calculate_covXZ(_zhat, _zmean, __sigpts, this->_X, wtsc);
	cv::Mat _invzcov;	cv::invert(_zcov, _invzcov, cv::DECOMP_CHOLESKY);
	// compute kalman gain
	cv::Mat _kalmangain =_xzcov * _invzcov ;
	this->_X = this->_correct_constraints(this->_X + _kalmangain * ( z - _zmean ));
	// update state covariance matrix
	this->_P = this->_P - (_kalmangain * _zcov * _kalmangain.t());
	this->_protectP();
}
// register functions
void SPKF::register_transition_func(TransitionFunc func)
{
	this->_statetransition = func;
}
int SPKF::register_obsrvfunc(ObservationFunc func, const cv::Mat &R)
{
	CV_Assert(R.type() == CV_64FC1);
	this->_obsrvfunc.push_back(func);
	this->_R.push_back(R);
	return (this->_obsrvfunc.size() - 1);
}
void SPKF::unregister_obsrvfunc(const int idx)
{
	if ( (idx < this->_obsrvfunc.size()) && (idx >= 0) ){
		this->_obsrvfunc.at(idx) = NULL;
		this->_R.at(idx) = cv::Mat();
	}
}
void SPKF::unregister_obsrvall()
{
	this->_obsrvfunc.clear();
	this->_R.clear();
}
// some non-important public functions
void SPKF::set_iteration_criteria(const double convergence, const double max_iter)
{
	this->_convergence = convergence;
	this->_max_iter = max_iter;
}
void SPKF::setQ(const cv::Mat& Q)
{
	CV_Assert(Q.rows == this->Lx);
	CV_Assert(Q.cols == this->Lx);
	CV_Assert(Q.type() == CV_64FC1);
	this->_Q = Q.clone();
}
void SPKF::setQDiagonal(const cv::Mat& _diag)
{
	CV_Assert(_diag.rows == this->Lx);
	CV_Assert(_diag.cols == 1);
	CV_Assert(_diag.type() == CV_64FC1);
	this->_Q = cv::Mat::diag(_diag);
}
void SPKF::setX(const cv::Mat& x)
{
	CV_Assert(x.rows == this->Lx);
	CV_Assert(x.cols == 1);
	CV_Assert(x.type() == CV_64FC1);
	this->_X = this->_correct_constraints(x);
}
cv::Mat SPKF::getP() const
{
	return this->_P.clone();
}
cv::Mat SPKF::getQ() const
{
	return this->_Q.clone();
}
cv::Mat SPKF::getX() const
{
	return this->_X.clone();
}
int SPKF::_getLx() const
{
	return this->Lx;
}
// to calculate lambda, weights, etc.
cv::Mat SPKF::matrix_squareroot(const cv::Mat& square_matrix)
{
	cv::Mat u, s, sm , vt, sqrtm;
	cv::SVD::compute(square_matrix, s, u, vt);
	cv::pow(s, 0.5, s);
	sm = cv::Mat::diag(s);	
	sqrtm = vt.t() * sm * vt;
	return sqrtm;
}
cv::Mat SPKF::calculate_sigmamat(const cv::Mat& P, const double lambda, const int L, const bool cholesky)
{
	cv::Mat _sqmat;
	cv::Mat _pmat = P * ( lambda + double(L));
	if ( cholesky ){
		_sqmat = SPKF::CholeskyDecomposition(_pmat);
	} else {
		_sqmat = SPKF::matrix_squareroot(_pmat);
	}
	return _sqmat;	
}
cv::Mat SPKF::calculate_sigmapts(const cv::Mat &xmean, const cv::Mat &sigmamat)
{
	int L = sigmamat.rows;
	cv::Mat _sigpts(L, 2*L + 1, CV_64FC1);
	for ( int _i = 0; _i < L; ++_i){
		_sigpts.at<double>(_i, 0) = xmean.at<double>(_i);
		_sigpts.col(_i + 1) = xmean + sigmamat.col(_i);
		_sigpts.col(_i + L + 1) = xmean - sigmamat.col(_i);
	}
	return _sigpts;
}
double SPKF::calculate_lambda(const unsigned int L, const double alpha, const double k)
{
	double keta;
	double lambda;
	if ( k < 0 ) keta = -k;
	else keta = k;
	lambda = (( alpha * alpha) * ( L + keta)) - (double)(L);
	return lambda;
}
cv::Mat SPKF::calculate_weights_covariance(const unsigned int L, const double lambda, const double alpha, const double beta)
{	
	double wt = 0.5 / ( lambda + double(L));
	cv::Mat weights(2*L+1, 1, CV_64FC1, cv::Scalar(wt));
	*(double*)(weights.data) = (lambda / ( lambda + float(L))) + ( 1. - (alpha * alpha) + beta);
	return weights;
}
cv::Mat SPKF::calculate_weights_mean(const unsigned int L, const double lambda)
{
	double wt = 0.5 / ( lambda + double(L));
	cv::Mat weights(2*L+1, 1, CV_64FC1, cv::Scalar(wt));
	*(double*)(weights.data) = (lambda / ( lambda + float(L)));

	return weights;
}
cv::Mat SPKF::calculate_Zmean(const cv::Mat &Z, const cv::Mat &wtsm)
{
	CV_Assert(Z.depth() == CV_64F );
	cv::Mat Zmean = cv::Mat::zeros(Z.rows, 1, CV_64FC1);	
	int elemstep = 8;
	int Zrowstep = Z.step / sizeof(Z.ptr()[0]);
	for ( int _zr = 0, _wr = 0 ; _zr < Z.rows; ++_zr ){
		int _curRow = _zr * Zrowstep;
		double sum = 0.;
		for ( int _wr = 0, i = 0; _wr < Zrowstep; _wr += elemstep , ++i){
			sum += (*(double*)(wtsm.data + _wr) * *(double*)(Z.data + _curRow + _wr));
		}
		*(double*)(Zmean.data + (elemstep*_zr)) = sum;
	}
	return Zmean;
}
cv::Mat SPKF::calculate_covZ(const cv::Mat &Z, const cv::Mat &Zmean, const cv::Mat &wtsc)
{
	CV_Assert(wtsc.rows == Z.cols);
	cv::Mat covZ = cv::Mat::zeros(Z.rows, Z.rows, CV_64FC1);	
	cv::Mat Zp(Z.rows, Z.cols, CV_64FC1);
	for ( int _i = 0; _i < Z.cols; ++_i ){
		Zp.col(_i) = Z.col(_i) -  Zmean; 
	}
	int elemstep = 8;
	int Zpcolstep = Zp.step;
	int CZcolstep = covZ.step;
	
	for ( int _r = 0; _r < Z.rows; ++_r ){
		int Zprrstep = Zpcolstep * _r;
		for ( int _c = _r; _c < Z.rows; ++_c ){
			int Zpccstep = Zpcolstep * _c;
			int Zcuritem = (CZcolstep * _r) + (_c * elemstep);
			int Zcuritemt = (CZcolstep * _c) + (_r * elemstep);
			for ( int _v = 0; _v < Zpcolstep; _v+=elemstep){
				*(double*)(covZ.data + Zcuritem) += *(double*)(wtsc.data + _v) * *(double*)(Zp.data + Zprrstep + _v) *
													*(double*)(Zp.data + Zpccstep + _v);
			}
			if ( _r != _c ){
				*(double*)(covZ.data + Zcuritemt) = *(double*)(covZ.data + Zcuritem);
			}
		}
	}
	return covZ;
}
cv::Mat SPKF::calculate_covXZ(const cv::Mat &Z, const cv::Mat &Zmean, const cv::Mat &sig_pts, const cv::Mat &xmean, const cv::Mat &wtsc)
{
	CV_Assert(Z.cols == sig_pts.cols);	
	cv::Mat covXZ = cv::Mat::zeros(sig_pts.rows, Z.rows, CV_64FC1);	
	cv::Mat Zp(Z.rows, Z.cols, CV_64FC1);
	cv::Mat Xp(sig_pts.rows, sig_pts.cols, CV_64FC1);
	
	for ( int _i = 0; _i < Z.cols; ++_i ){
		Zp.col(_i) = Z.col(_i) -  Zmean; 
		Xp.col(_i) = sig_pts.col(_i) - xmean;
	}
	
	for ( int _r = 0; _r < sig_pts.rows; ++_r ){
		for ( int _c = 0; _c < Z.rows; ++_c ){
			for ( int _v = 0; _v < wtsc.rows; ++_v ){
				covXZ.at<double>(_r,_c) += wtsc.at<double>(_v) * Xp.at<double>(_r,_v) * Zp.at<double>(_c,_v) ;
			}
		}
	}
	return covXZ;
}
cv::Mat SPKF::CholeskyDecomposition(const cv::Mat& input)
{
	CV_Assert(input.depth() == CV_32F || input.depth() == CV_64F );
	CV_Assert(input.rows == input.cols );
	cv::Mat _result = input.clone();	 
	int rowstep = _result.step / sizeof(_result.ptr()[0]);	
	if  ( input.depth() == CV_32F ){
		cv::Cholesky( _result.ptr<float>(), rowstep, _result.cols, 0, 0, 0);		
		for ( int i = 0; i < _result.cols; ++i ){
			int _curItem = 4 * i;
			int _curRow = rowstep * i;
			float *cur = (float*)(_result.data + _curRow + _curItem);
			if ( *cur > 1. ) *cur = float(1. / *cur);
			for ( _curItem += 4; _curItem < rowstep  ; _curItem += 4 )
				*(float*)(_result.data + _curRow + _curItem) = 0.;		
		}
	}
	if  ( input.depth() == CV_64F ){
		cv::Cholesky( _result.ptr<double>(), rowstep, input.cols, 0, 0, 0);
		for ( int i = 0; i < _result.cols; ++i ){
			int _curItem = 8 * i;
			int _curRow = rowstep * i;
			double *cur = (double*)(_result.data + _curRow + _curItem);
			if ( *cur > 1. ) *cur = double(1. / *cur);
			for ( _curItem += 8; _curItem < rowstep  ; _curItem += 8 )
				*(double*)(_result.data + _curRow + _curItem) = 0.;		
		}		
	}	
	return _result;
}
// batch function apply
cv::Mat SPKF::_cvectorapply_observationfunc(const cv::Mat& xcvec, const int idx, void* ptr, const cv::Mat &noise) const
{
	CV_Assert(xcvec.rows > 0);
	CV_Assert(xcvec.cols > 0);
	ObservationFunc obfun = this->_obsrvfunc.at(idx);
	cv::Mat _tempcol;
	cv::Mat _zhat(this->_R.at(idx).rows, xcvec.cols, CV_64FC1);
	// propogate through function
	for ( int _i = 0; _i < xcvec.cols; ++_i ){
		_tempcol = obfun(xcvec.col(_i), ptr, noise.col(_i));
		for ( int _k = 0; _k < _zhat.rows; ++_k){
			_zhat.at<double>(_k,_i) = _tempcol.at<double>(_k);
		}
	}
	return _zhat;
}
cv::Mat SPKF::_cvectorapply_predictionfunc(const cv::Mat& xcvec, const double delt, void* ptr, const cv::Mat &noise) const
{
	CV_Assert(xcvec.rows > 0);
	CV_Assert(xcvec.cols > 0);
	cv::Mat _xhat(xcvec.rows, xcvec.cols, CV_64FC1);
	cv::Mat _tempcol;
	// propogate through function
	for ( int _i = 0; _i < xcvec.cols; ++_i){
		_tempcol = this->_correct_constraints(this->_statetransition(xcvec.col(_i), delt, ptr, noise.col(_i)));
		for ( int _k = 0; _k < xcvec.rows; ++_k){
			_xhat.at<double>(_k,_i) = _tempcol.at<double>(_k);
		}
	}
	return _xhat;
}
cv::Mat SPKF::_correct_constraints(const cv::Mat& mat) const
{
	// DO Nothing here
	cv::Mat _res;	mat.copyTo(_res);
	return _res;
}
cv::Mat SPKF::_batchcorrect_constraints(const cv::Mat& mat) const
{
	cv::Mat _col1, _corrected(mat.rows, mat.cols, CV_64FC1);
	for ( int _i = 0; _i < mat.cols; ++_i ){
		_col1 = this->_correct_constraints(mat.col(_i));
		for ( int _j = 0; _j < mat.rows; ++_j){
			_corrected.at<double>(_j, _i) = _col1.at<double>(_j);
		}
	}
	return _corrected;
}
void SPKF::_protectP()
{
	bool prob_detected = false;
	for ( int i = 0; i < this->_P.rows; ++i ){
		for ( int j = 0; j < this->_P.cols; ++j ){
			if ( this->_P.at<double>(i, j) > 10. ){
				fprintf(stderr, "P matrix element becoming larger than 10, resetting P\n");
				prob_detected = true;
			}
			if ( this->_P.at<double>(i, j) < -1E2 ){
				fprintf(stderr, "P matrix element becoming negative resetting P\n");
				prob_detected = true;
			}
		}
	}
	if ( prob_detected ){
		this->_P = this->_Q.clone();
	}
}
// constructors and destructors
SPKF::SPKF(const unsigned int dimX, const double _alpha, const double _beta, const double _k)
{
	this->Lx = dimX;
	this->alpha = _alpha;
	this->beta = _beta;
	this->k = _k;
	this->_convergence = 1E-3;
	this->_max_iter = 500;
	this->_X = cv::Mat::zeros(this->Lx, 1, CV_64FC1);
	this->_P = cv::Mat::eye(this->Lx, this->Lx, CV_64FC1);
	this->_Q = cv::Mat::diag(cv::Mat::ones(this->Lx, 1, CV_64FC1));
}

SPKF::SPKF(const SPKF& other)
{
	this->Lx = other.Lx;
	this->alpha = other.alpha;
	this->beta = other.beta;
	this->k = other.k;
	this->_convergence = other._convergence;
	this->_max_iter = other._max_iter;
	this->_X = other._X.clone();
	this->_Q = other._Q.clone();
	this->_statetransition = other._statetransition;
	this->_obsrvfunc = other._obsrvfunc;
	this->_P = other._P.clone();
	this->_R = other._R;
}
SPKF::~SPKF() { }