template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::operator *= (const TYPE val){
	scale *= val;
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::operator /= (const TYPE val){
	if(CUDA){
		cuWrap::scale(dataPrt(), TYPE(1)/val, size());
	}else{
		eMat() /= val;
	}
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator + (const TYPE val) const{
	MatriX< TYPE,CUDA> tmp(*this, STRU);
	tmp = val;
	return *this + tmp;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::operator += (const TYPE val){
	MatriX<TYPE, CUDA> tmp(*this, STRU);
	tmp = val;
	*this += tmp;
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::operator -= (const TYPE val){
	MatriX< TYPE,CUDA> tmp(*this, STRU);
	tmp = val;
	*this -= tmp;
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator - (const TYPE val) const{
	MatriX< TYPE,CUDA> tmp(*this, STRU);
	tmp = val;
	return *this - tmp;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator * (const MatriX<TYPE, CUDA> &mat) const{
	int rRow = rows();
	int rCol = mat.cols();
	int joint = cols();
	TYPE mScale = mat.scale * scale;
	MatriX<TYPE,CUDA> ret(rRow, rCol);
	if(mat.rows()!= joint){
		Assert("矩阵维度不匹配,无法进行乘法运算");	
	}
	if(CUDA){
		cuWrap::gemm(trans(), mat.trans(), rRow, rCol, joint, dataPrt(), mat.dataPrt(), ret.dataPrt(), mScale);
	}else{
		if(!trans()){
			if(!mat.trans()){
				ret.eMat() = eMat() * mat.eMat() * mScale;				
			}else{
				ret.eMat() = eMat() * mat.eMat().transpose()  * mScale;
			}
		}else{
			if(!mat.trans()){
				ret.eMat() = eMat().transpose() * mat.eMat()  * mScale;
			}else{
				ret.eMat() = eMat().transpose() * mat.eMat().transpose()  * mScale;
			}
		}
	}
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::cwiseProduct (const MatriX<TYPE, CUDA> &mat) const{
	if(mat.size() != size()){
		Assert("矩阵大小不匹配,无法进行cwiseProduct运算");	
	}
	MatriX<TYPE, CUDA> ret(*this, STRU);
	if(CUDA){
			if(trans() == mat.trans()){
				cuWrap::product( mat.dataPrt(), dataPrt(), ret.dataPrt(), size());
			}else{
				MatriX<TYPE, CUDA> tmp(mat, TURN);
				cuWrap::product( tmp.dataPrt(), dataPrt(), ret.dataPrt(), size());
			}
	}else{
		if(trans() == mat.trans()){
			ret.eMat() =  eMat().cwiseProduct(mat.eMat());			
		}else{
			ret.eMat() =  eMat().cwiseProduct(mat.eMat().transpose());				
		}
	}
	ret.scale =  mat.scale * scale;	
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::cwiseQuotient (const MatriX<TYPE, CUDA> &mat) const{
	if(mat.size() != size()){
		Assert("矩阵大小不匹配,无法进行cwiseProduct运算");	
	}
	MatriX<TYPE, CUDA> ret(mat, SCL);
	if(CUDA){
		if(trans() == mat.trans()){
			cuWrap::quotient(dataPrt(),  ret.dataPrt(), ret.dataPrt(), size());
		}else{			
			MatriX<TYPE, CUDA> tmp(*this, TURN);
			cuWrap::quotient(tmp.dataPrt(), ret.dataPrt(), ret.dataPrt(), size());
		}
	}else{
		if(trans() == mat.trans()){
			ret.eMat() =  eMat().cwiseQuotient(ret.eMat());			
		}else{
			ret.eMat() =  eMat().cwiseQuotient(ret.eMat().transpose()).transpose();				
		}
	}
	ret.scale =   scale ;
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::cwiseInverse () const{
	MatriX<TYPE, CUDA> ret(*this, SCL);
	if(CUDA){
		cuWrap::cwiseInverse(ret.dataPrt(), ret.dataPrt(), size());
	}else{
		ret.eMat() =  ret.eMat().cwiseInverse();					
	}
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::rankUpdate() const{
	MatriX<TYPE, CUDA> ret = MatriX< TYPE, CUDA>::Zero(rows(), rows());
	if(CUDA){
	//	cuWrap::rankUpdate(ret.dataPrt(), dataPrt());
	}else{
		ret.eMat() = ret.eMat().selfadjointView<Eigen::Upper>().rankUpdate(eMat());
	}
	ret.scale = scale * scale;
	return ret;
}
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::dot(const MatriX<TYPE, CUDA> &mat) const{
	MatriX<TYPE, CUDA> tmp = this->cwiseProduct(mat);
	return tmp.allSum();
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator + (const MatriX<TYPE, CUDA> &mat) const{
	if(scale == 0){
		MatriX<TYPE,CUDA> ret = mat;
		return ret;
	}
	MatriX<TYPE,CUDA> ret = *this;
	ret += mat;
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator - (const MatriX<TYPE, CUDA> &mat) const{
	if(scale == 0){
		MatriX<TYPE,CUDA> ret = mat;
		ret *= -1;
		return ret;
	}
	MatriX<TYPE,CUDA> ret = *this;
	ret -= mat;
	return ret;
}

template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA>& MatriX< TYPE,CUDA>::operator += (const MatriX<TYPE,CUDA> &mat){
	if(mat.scale == 0){
		return *this;
	}
	copyRealise(false, false);
	if(matPlusVec(mat)){
		return *this;
	}
	if(CUDA){
		cuWrap::plus(false, mat.trans()!= trans(), realRows(), realCols(), dataPrt(), mat.dataPrt(), dataPrt(), scale, mat.scale);
	}else{
		if(trans() == mat.trans()){
				eMat() = eMat() * scale +  mat.eMat() * mat.scale;			
		}else{
				eMat() = eMat() * scale + mat.eMat().transpose() * mat.scale;
		}
	}
	scale = 1;
	return *this;
};
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA>& MatriX< TYPE,CUDA>::add(const MatriX<float,CUDA> &mat){
	if(sizeof(TYPE) == sizeof(float)){
		*this += mat;
		return *this;
	}else{
		if(CUDA){
			if(trans() == mat.trans()){
				cuWrap::plusFloatMat(dataPrt(), scale, mat.dataPrt(), mat.scale, size());
			}else{
				MatriX<float, CUDA> tmp = mat;
				tmp.transMem();
				cuWrap::plusFloatMat(dataPrt(), scale, tmp.dataPrt(), tmp.scale, size());
			}
		}else{
			if(trans() == mat.trans()){
				std::transform(dataPrt(), dataPrt() + size(), mat.dataPrt(), dataPrt(), cpu_funcs::plusFloatMat(scale, mat.scale));				
			}else{
				MatriX<float, CUDA> tmp = mat;
				tmp.transMem();
				std::transform(dataPrt(), dataPrt() + size(), tmp.dataPrt(), dataPrt(), cpu_funcs::plusFloatMat(scale, tmp.scale));	
			}

		}
		scale = 1;
		return *this;
	}
	
};
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA>& MatriX< TYPE,CUDA>::add(const MatriX<double,CUDA> &mat){
	MatriX< TYPE, CUDA> tmp = mat;
	*this += tmp;
	return *this;
};
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA>& MatriX< TYPE,CUDA>::operator -= (const MatriX<TYPE,CUDA> &mat){
	*this += (-mat);
	return *this;
};
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator * (const TYPE val) const{
	MatriX<TYPE,CUDA> ret = *this;
	ret *= val;
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator / (const TYPE val) const{
	MatriX<TYPE,CUDA> ret = *this;
	ret /= val;
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> operator * (TYPE val ,const MatriX<TYPE,CUDA> &m){
	MatriX<TYPE,CUDA> ret = m;
	ret *= val;
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> operator + (TYPE val ,const MatriX<TYPE,CUDA> &m){
	MatriX<TYPE,CUDA> ret = m;
	ret = val;
	return ret + m;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> operator - (TYPE val ,const MatriX<TYPE,CUDA> &m){
	MatriX<TYPE,CUDA>  ret = m;
	ret = val;
	return ret - m;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> operator -(const MatriX<TYPE,CUDA> &m){
	MatriX<TYPE,CUDA> ret = m * (-1);
	return ret;
}

template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::T() const{
	MatriX<TYPE,CUDA> ret = *this;
	ret.setTrans();
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::transpose() const{
	return T();
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::inverse()  const{
	return inv();
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::inv() const{
	MatriX<TYPE, CUDA> ret(*this, SCL);
	if(CUDA){		
		eigenMat tmp(realRows(), realCols());
		cuWrap::memD2H(tmp.data(), ret.dataPrt(), sizeof(TYPE) * size());
		tmp = tmp.inverse();
		cuWrap::memH2D(ret.dataPrt(), tmp.data(), sizeof(TYPE) * size());	
	}else{
		ret.eMat() = ret.eMat().inverse();	
	}
	return ret;

};
