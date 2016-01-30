template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::tanh(){
	copyRealise(true, false);
	if(CUDA){
		cuWrap::tanh(dataPrt(), size());
	}else{
		std::transform(dataPrt(), dataPrt() + size(), dataPrt(), cpu_funcs::tanh<TYPE>);
	}
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> &MatriX< TYPE,CUDA>::sigm(){
	copyRealise(true, false);
	if(CUDA){
		cuWrap::sigm(dataPrt(), size());
	}else{
		std::transform(dataPrt(), dataPrt() + size(), dataPrt(), cpu_funcs::sigm<TYPE>);
	}
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> &MatriX< TYPE,CUDA>::square(){
	copyRealise(true, false);
	if(CUDA){
		cuWrap::square(dataPrt(), size());
	}else{
		std::transform(dataPrt(), dataPrt() + size(), dataPrt(), cpu_funcs::square<TYPE>);
	}
	return *this;
};
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> &MatriX< TYPE,CUDA>::abs(){
	copyRealise(false, false);
	if(CUDA){
		cuWrap::abs(dataPrt(), size());
	}else{
		std::transform(dataPrt(), dataPrt() + size(), dataPrt(), cpu_funcs::abs<TYPE>);
	}
	scale = ::abs(scale);
	return *this;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> sigm (const MatriX<TYPE, CUDA> &m){
	MatriX<TYPE, CUDA> ret = m;	
	return ret.sigm();
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA>  tanh (const MatriX<TYPE, CUDA> &m){
	MatriX<TYPE, CUDA> ret = m;		
	return ret.tanh();
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> square (const MatriX<TYPE, CUDA> &m){
	MatriX<TYPE, CUDA> ret = m;	
	return ret.square();
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA> abs (const MatriX<TYPE, CUDA> &m){
	MatriX<TYPE, CUDA> ret = m;	
	return ret.abs();
};
