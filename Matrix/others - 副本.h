
template <typename TYPE, bool CUDA>
int MatriX< TYPE, CUDA>::elementPos(int _row, int _col){
	return rows() * _col + _row; 	
}
template <typename TYPE, bool CUDA>
int MatriX< TYPE, CUDA>::realPos(int idx){
	if(!transFlag){
		return idx; 
	}else{
		int _col = idx / rows();
		int _row = idx % cols();
		return cols() * _col + _row; 
	}
}

template <typename TYPE, bool CUDA>
bool MatriX< TYPE, CUDA>::is_CUDA() const{
	return CUDA;
}
template <typename TYPE, bool CUDA>
TYPE * MatriX< TYPE, CUDA>::data() {
	copy_realised(true);
	if(transFlag){
		real_transpose();
	}
	return prt->body;
}



template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> &MatriX< TYPE,CUDA>::tanh(){
	copyRealise(true);
	if(CUDA){
		cuWrap::tanh(data(), size);
	}else{
		std::transform(data(), data() + size, data(), cpu_funcs::tanh<TYPE>);
	}
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> &MatriX< TYPE,CUDA>::sigm(){
	copyRealise(true);
	if(CUDA){
		cuWrap::tanh(data(), size);
	}else{
		std::transform(data(), data() + size, data(), cpu_funcs::sigm<TYPE>);
	}
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> &MatriX< TYPE,CUDA>::square(){
	copyRealise(true);
	if(CUDA){
		cuWrap::tanh(data(), size);
	}else{
		std::transform(data(), data() + size, data(), cpu_funcs::square<TYPE>);
	}
	scale *= scale;
	return *this;
};
template <typename TYPE, bool CUDA>
inline MatriX<TYPE,CUDA> sigm (const MatriX<TYPE,CUDA> &m){
	MatriX<TYPE,CUDA> ret(m, QUO);	
	return ret.sigm();
};
template <typename TYPE, bool CUDA>
inline MatriX<TYPE,CUDA> tanh (const MatriX<TYPE,CUDA> &m){
	MatriX<TYPE,CUDA> ret = m;	
	return ret.tanh();
};
template <typename TYPE, bool CUDA>
inline MatriX<TYPE,CUDA> square (const MatriX<TYPE,CUDA> &m){
	MatriX<TYPE,CUDA> ret = m;	
	return ret.square();
};
template <typename TYPE, bool CUDA>
string MatriX< TYPE,CUDA>::structure() const{
	stringstream ret;
	ret<<"\ncols:"<<cols<<" rows:"<<rows<<" size:"<<size<<" memery:"<<size*sizeof(TYPE)<<" prt_count:"<<prt->count<<"\n";
	ret<<"cuda:"<<CUDA<<" data_type:"<<typeid(TYPE).name()<<" trans:"<<transFlag<<" scale:"<<scale<<"\n";
	return ret.str();	
};
template <typename TYPE, bool CUDA>
ostream& operator <<(ostream &os, const MatriX<TYPE, CUDA> &m){
	eigenMat s(m.rows(), m.cols());
	os<<"\n";
	if(CUDA){
		MatriX<TYPE,CUDA> tmp(m, QUO);
		cuWrap::memD2H(s.dataPrt(), tmp.dataPrt(), sizeof(TYPE) * m.size());
	}else{
		s = m.eMat();
	}
	if((m.transFlag){
		s = s.transpose();
	}
	s *= m.scale;
	os<<s<<"\n";
	return os;
}
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> & MatriX< TYPE,CUDA>::load_mat(TYPE * src){
	copy_realised(false);
	scale = 1;
	prt->copy_data(src, false);
	return *this;
};
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::sum(){
	if(CUDA){
		return cuWrap::sum(prt->body, size) * scale;		
	}else{
		Matrix< TYPE,-1, -1> &t = *(prt->eigen_mat);
		return t.array().sum() * scale;
	}
}; 
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::norm(){
	if(CUDA){
		return cuWrap::norm(prt->body, size) * scale;		
	}else{
		Matrix< TYPE,-1, -1> &t = *(prt->eigen_mat);
		return t.norm() * scale;
	}
}; 
template <typename TYPE, bool CUDA>
TYPE MatriX< TYPE,CUDA>::squaredNorm(){
	TYPE ret = (*this).norm();
	return ret * ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX< TYPE,CUDA>::colSum(){
	MatriX< TYPE,CUDA> ret(1, cols());
	if(CUDA){
		cuWrap::colSum(ret.prt->body, prt->body, scale, transFlag, rows(), cols());		
	}else{		
		Matrix< TYPE,-1, -1> &t = *(prt->eigen_mat);
		if(transFlag){
			//		ret.prt->copy_data((t.colise().sum()).data(),false);
		}else{
			//		ret.prt->copy_data((t.rowwise().sum()).data(),false);
		}
		ret *= scale;
	}
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX< TYPE,CUDA>::rowSum(){
	MatriX< TYPE,CUDA> ret(rows(), 1);
	if(CUDA){
		cuWrap::rowSum(ret.prt->body, prt->body, scale, transFlag, rows(), cols());		
	}else{		
		Matrix< TYPE,-1, -1> &t = *(prt->eigen_mat);
		if(transFlag){
			//			ret.prt->copy_data((t.rowwise().sum()).data(),false);
		}else{
			//			ret.prt->copy_data((t.colwise().sum()).data(),false);
		}
		ret *= scale;
	}
	return ret;
};

/*

template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::mean(){
	int n = cols();
	MatriX<TYPE,CUDA> ret(n);
	for(int i = 0; i<n; i++){

	}
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX<TYPE,CUDA>::MSE(){
};
*/