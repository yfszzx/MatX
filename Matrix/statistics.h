template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::allSum() const{
	if(CUDA){
		return cuWrap::sum(dataPrt(), size()) * scale;		
	}else{
		return std::accumulate(dataPrt(), dataPrt() + size(), (TYPE)0) * scale;
	}
};
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::allMean()  const{
	return allSum()/size();
};
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::allMSE()  const{
	allMean();
	return ::square(*this - allMean()).allMean();
};
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::allMSE(TYPE & avg)  const{
	avg = allMean();
	return ::square(*this - avg).allMean();
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX< TYPE,CUDA>::colsSum() const{
	MatriX< TYPE,CUDA> ret(rows(), 1);
	if(CUDA){
		cuWrap::colSum(ret.dataPrt(), dataPrt(), scale, trans(), rows(), cols());		
	}else{		
		eigenMat ones = eigenMat::Ones(cols(), 1);
		if(trans()){
			ret.eMat() = eMat().transpose() * ones * scale;
		}else{		
			ret.eMat() = eMat() * ones * scale;
		}
	}
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX< TYPE,CUDA>::sum() const{
	if(cols() == 1 || rows() == 1){
		MatriX< TYPE,CUDA> ret(1, 1);
		ret.assignment(0,0,allSum());
		return ret;
	}
	return rowsSum();
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX< TYPE,CUDA>::rowsSum() const{
	MatriX< TYPE,CUDA> ret(1, cols());

	if(CUDA){
		cuWrap::rowSum(ret.dataPrt(), dataPrt(), scale, trans(), rows(), cols());		
	}else{		
		eigenMat ones = eigenMat::Ones(1, rows());
		if(trans()){
			ret.eMat() = ones * eMat().transpose() * scale;
		}else{		
			ret.eMat() = ones * eMat() * scale;
		}
	}
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::mean() const{
	return sum()/rows();
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::MSE() const{
	return ::square(*this - mean()).mean();
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::RMS( MatriX<TYPE,CUDA> & avg) const{
	avg = mean();
	return ::sqrt(::square(*this - avg).mean());
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::RMS() const{
	return ::sqrt(::square(*this - mean()).mean());
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::MSE( MatriX<TYPE,CUDA> & avg) const{
	avg = mean();
	return ::square(*this - avg).mean();
};
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::norm() const{
	if(CUDA){
		return cuWrap::norm(dataPrt(), size()) * scale;		
	}else{
		return eMat().norm() * scale;
	}
}; 
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::squaredNorm() const{
	TYPE ret = this->norm();
	return ret * ret;
};
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::norm2() const{
	TYPE ret = this->norm();
	return ret * ret;
};
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::correl(MatriX<TYPE, CUDA>& mat) const{
	int n = mat.size();
	if(size()!= n){
		Assert("矩阵维度不一致，无法计算相关系数");
	}
	double xsum = allSum();
	double ysum = mat.allSum();
	return (n * this->dot(mat) - xsum * ysum)/::sqrt(( n * this->squaredNorm() - xsum * xsum) *( n * mat.squaredNorm() - ysum * ysum));
};
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::allMin() const{
	TYPE ret;
	if(CUDA){
		if(scale > 0){
			ret = cuWrap::min_element(dataPrt(), size());
		}else{
			ret = cuWrap::max_element(dataPrt(), size());
		}
	}else{
		if(scale > 0){
			ret = *min_element(dataPrt(), dataPrt() + size());
		}else{
			ret = *max_element(dataPrt(), dataPrt() + size());
		}
	}
	return ret * scale;
};
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::allMax() const{
	TYPE ret;
	if(CUDA){
		if(scale > 0){
			ret = cuWrap::max_element(dataPrt(), size());
		}else{
			ret = cuWrap::min_element(dataPrt(), size());
		}
	}else{
		if(scale > 0){
			ret = *max_element(dataPrt(), dataPrt() + size());
		}else{
			ret = *min_element(dataPrt(), dataPrt() + size());
		}
	}
	return ret * scale;
};

