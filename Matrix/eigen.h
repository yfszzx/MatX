template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::eigenValues() const{
	if(rows() != cols()){
		Assert("�����Ƿ��󣬲����������ֵ");
	}
	MatriX<TYPE, CUDA> ret(rows());
	MatriX<TYPE, CUDA> tmp(*this);
	tmp.copyRealise(true, true);
	SelfAdjointEigenSolver<eigenMat> es(tmp.eMat()); 
	if (es.info() != Eigen::Success) {  
		cout<<"\n�������ֵʧ��";
		return MatriX<TYPE, CUDA>::Zero(0,0);
	}
	ret.loadMat(es.eigenvalues().data());
	return ret;

};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> MatriX<TYPE, CUDA>::eigenSolver( MatriX<TYPE, CUDA>& eigenVals) const{
	if(rows() != cols()){
		Assert("�����Ƿ��󣬲����������ֵ");
	}
	MatriX<TYPE, CUDA> ret(rows(),rows());
	MatriX<TYPE, CUDA> tmp(*this);
	tmp.copyRealise(true, true);
	SelfAdjointEigenSolver<eigenMat> es(tmp.eMat()); 
	if (es.info() != Eigen::Success) {  
		cout<<"\n�������ֵʧ��";
		eigenVals = MatriX<TYPE, CUDA>::Zero(0);

	}else{
		eigenVals = MatriX<TYPE, CUDA>::Zero(rows());
	}
	eigenVals.loadMat(es.eigenvalues().data());
	ret.loadMat(es.eigenvectors().data());
	return ret;
};
template <typename TYPE, bool CUDA>
TYPE MatriX<TYPE, CUDA>::spectralRadius() const{
	if(rows() != cols()){
		Assert("�����Ƿ��󣬲������װ뾶");
	}
	MatriX<TYPE, CUDA> t = this->eigenValues();
	return this->eigenValues().abs().allMax();
};
