template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::operator *= (const TYPE val){
	scale *= val;
	if(fix()){
		copyRealise(true);
	}
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::operator /= (const TYPE val){
	scale /= val;
	if(fix()){
		copyRealise(true);
	}
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator + (const TYPE val) const{
	MatriX< TYPE,CUDA> tmp(*this, STR);
	tmp = val;
	return *this + tmp;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::operator += (const TYPE val){
	MatriX<TYPE, CUDA> tmp(*this, STR);
	tmp = val;
	*this += tmp;
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::operator -= (const TYPE val){
	MatriX< TYPE,CUDA> tmp(*this, STR);
	tmp = val;
	*this -= tmp;
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator - (const TYPE val) const{
	MatriX< TYPE,CUDA> tmp(*this, STR);
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
	MatriX<TYPE, CUDA> ret(mat, STR);
	if(CUDA){
			if(trans() == mat.trans()){
				cuWrap::product( mat.dataPrt(), dataPrt(), ret.dataPrt(), size());
			}else{
				MatriX<TYPE, CUDA> tmp(mat.transpose(), TRN);
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
	ret.setTrans(trans());
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::cwiseQuotient (const MatriX<TYPE, CUDA> &mat) const{
	if(mat.size() != size()){
		Assert("矩阵大小不匹配,无法进行cwiseProduct运算");	
	}
	MatriX<TYPE, CUDA> ret(mat, STR);
	if(CUDA){
		if(trans() == mat.trans()){
			cuWrap::quotient( mat.dataPrt(), dataPrt(), ret.dataPrt(), size());
		}else{
			MatriX<TYPE, CUDA> tmp(mat.T(), TRN);
			cuWrap::quotient( tmp.dataPrt(), dataPrt(), ret.dataPrt(), size());
		}
	}else{
		if(trans() == mat.trans()){
			ret.eMat() =  eMat().cwiseQuotient(mat.eMat());			
		}else{
			ret.eMat() =  eMat().cwiseQuotient(mat.eMat().transpose());				
		}
	}
	ret.scale =   scale / mat.scale;
	ret.setTrans(trans());
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::cwiseInverse () const{
	MatriX<TYPE, CUDA> ret(mat, STR);
	if(CUDA){
		cuWrap::cwiseInverse(ret.dataPrt(), dataPrt(), size());
	}else{
		ret.eMat() =  eMat().cwiseInverse();					
	}
	ret.scale = 1.0f/scale;
	ret.setTrans(trans());
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::rankUpdate() const{
	MatriX<TYPE, CUDA> ret = MatriX< TYPE, CUDA>::Zero(rows(), rows());
	if(CUDA){

	}else{
		ret.eMat() = ret.eMat().selfadjointView<Eigen::Upper>().rankUpdate(eMat());
	}
	ret.scale = scale * scale;
	return ret;
}
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::dot(const MatriX<TYPE, CUDA> &mat) const{
	MatriX<TYPE, CUDA> tmp(this->cwiseProduct(mat), QUO);
	return tmp.allSum();
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator + (const MatriX<TYPE, CUDA> &mat) const{
	if(scale == 0){
		MatriX<TYPE,CUDA> ret(mat, QUO);
		return ret;
	}
	MatriX<TYPE,CUDA> ret(*this, QUO);
	ret += mat;
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator - (const MatriX<TYPE, CUDA> &mat) const{
	if(scale == 0){
		MatriX<TYPE,CUDA> ret(mat, QUO);
		ret *= -1;
		return ret;
	}
	MatriX<TYPE,CUDA> ret(*this, QUO);
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
				MatriX<TYPE, CUDA> tmp(mat.T(), TRN);
				cuWrap::plusFloatMat(dataPrt(), scale, tmp.dataPrt(), tmp.scale, size());
			}
		}else{
			if(trans() == mat.trans()){
				std::transform(dataPrt(), dataPrt() + size(), mat.dataPrt(), dataPrt(), cpu_funcs::plusFloatMat(scale, mat.scale));				
			}else{
				MatriX<TYPE, CUDA> tmp(mat.T(), TRN);
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
	*this += mat;
	return *this;
};
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA>& MatriX< TYPE,CUDA>::operator -= (const MatriX<TYPE,CUDA> &mat){
	*this += (-mat);
	return *this;
};
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator * (const TYPE val) const{
	MatriX<TYPE,CUDA> ret(*this, QUO);
	ret *= val;
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> MatriX< TYPE,CUDA>::operator / (const TYPE val) const{
	MatriX<TYPE,CUDA> ret(*this, QUO);
	ret /= val;
	return ret;
}
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> operator * (TYPE val ,const MatriX<TYPE,CUDA> &m){
	MatriX<TYPE,CUDA> ret(m, MatriX< TYPE,CUDA>::QUO);
	ret *= val;
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> operator + (TYPE val ,const MatriX<TYPE,CUDA> &m){
	MatriX<TYPE,CUDA> ret(m, MatriX< TYPE,CUDA>::QUO);
	ret = val;
	return ret + m;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> operator - (TYPE val ,const MatriX<TYPE,CUDA> &m){
	MatriX<TYPE,CUDA>  ret(m, MatriX< TYPE,CUDA>::QUO);
	ret = val;
	return ret - m;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> operator -(const MatriX<TYPE,CUDA> &m){
	MatriX<TYPE,CUDA> ret(m * (-1), MatriX< TYPE,CUDA>::QUO);
	return ret;
}

template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::T() const{
	MatriX<TYPE,CUDA> ret(*this, QUO);
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
	MatriX<TYPE, CUDA> ret = *this;
	ret.copyRealise(false, false);
	if(!CUDA){		
		ret.eMat() = ret.eMat().inverse();		
	}else{
		eigenMat tmp(realRows(), realCols());
		cuWrap::memD2H(tmp.data(), dataPrt(), sizeof(TYPE) * size());
		tmp = tmp.inverse();
		cuWrap::memH2D(ret.dataPrt(), tmp.data(), sizeof(TYPE) * size());
	}
	ret. scale = 1/ret.scale;
	return ret;

};
