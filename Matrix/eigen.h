template <typename TYPE, bool CUDA>
MatriX<TYPE, false> MatriX<TYPE, CUDA>::eigenValues() const{
	if(rows() != cols()){
		Assert("�����Ƿ��󣬲����������ֵ");
	}
	MatriX<TYPE, false> ret(rows());
	MatriX<TYPE, false> tmp(*this);
	tmp.copyRealise(true, true);
	SelfAdjointEigenSolver<eigenMat> es(tmp.eMat()); 
	if (es.info() != Eigen::Success) {  
		cout<<"\n�������ֵʧ��";
		return MatriX<TYPE, false>::Zero(0,0);
	}
	ret.loadMat(es.eigenvalues().data());
	return ret;

};
template <typename TYPE, bool CUDA>
MatGroup<TYPE, CUDA> MatriX<TYPE, CUDA>::eigenSolver( MatriX<TYPE, false>& eigenVals) const{
	if(rows() != cols()){
		Assert("�����Ƿ��󣬲����������ֵ");
	}
	MatGroup<TYPE, CUDA> ret;

	MatriX<TYPE, false> tmp(*this);
	tmp.copyRealise(true, true);
	SelfAdjointEigenSolver<eigenMat> es(tmp.eMat()); 
	if (es.info() != Eigen::Success) {  
		cout<<"\n�������ֵʧ��";
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
		Assert("�����Ƿ��󣬲������װ뾶");
	}
	MatriX<TYPE, CUDA> t = this->eigenValues();
	return this->eigenValues().abs().allMax();
};