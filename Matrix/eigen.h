template <typename TYPE, bool CUDA>
MatriX<TYPE, false> MatriX<TYPE, CUDA>::eigenValues() const{
	if(rows() != cols()){
		Assert("矩阵不是方阵，不能求解特征值");
	}
	MatriX<TYPE, false> ret(rows());
	MatriX<TYPE, false> tmp(*this);
	tmp.copyRealise(true, true);
	SelfAdjointEigenSolver<eigenMat> es(tmp.eMat()); 
	if (es.info() != Eigen::Success) {  
		cout<<"\n求解特征值失败";
		return MatriX<TYPE, false>::Zero(0,0);
	}
	ret.loadMat(es.eigenvalues().data());
	return ret;

};
template <typename TYPE, bool CUDA>
MatGroup<TYPE, CUDA> MatriX<TYPE, CUDA>::eigenSolver( MatriX<TYPE, false>& eigenVals) const{
	if(rows() != cols()){
		Assert("矩阵不是方阵，不能求解特征值");
	}
	MatGroup<TYPE, CUDA> ret;

	MatriX<TYPE, false> tmp(*this);
	tmp.copyRealise(true, true);
	SelfAdjointEigenSolver<eigenMat> es(tmp.eMat()); 
	if (es.info() != Eigen::Success) {  
		cout<<"\n求解特征值失败";
		eigenVals = MatriX<TYPE, false>::Zero(0);

	}else{
		eigenVals = MatriX<TYPE, false>::Zero(rows());
	}

	memcpy(ret.dataPrt(), es.eigenvalues().data(), sizeof(TYPE) * rows());
	return ret;
};
template <typename TYPE, bool CUDA>
TYPE MatriX<TYPE, CUDA>::spectralRadius() const{
	if(rows() != cols()){
		Assert("矩阵不是方阵，不能求谱半径");
	}
	MatriX<TYPE, CUDA> t = this->eigenValues();
	return this->eigenValues().abs().allMax();
};