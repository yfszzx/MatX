template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA>  MatriX<TYPE, CUDA>::operator > (const MatriX<TYPE, CUDA> & m)  const{
	if(size() != m.size()){
		Assert("矩阵大小不一致，无法进行比较");
	}
	MatriX<TYPE, CUDA> ret(*this, STRU);
	if(trans() == m.trans()){
	if(CUDA){
		cuWrap::Gre(ret.dataPrt(), dataPrt(), m.dataPrt(), size());
	}else{
		std::transform(dataPrt(), dataPrt() + size(), m.dataPrt(), ret.dataPrt(), cpu_funcs::Gre<TYPE>);
	}
	}else{
		MatriX<TYPE, CUDA> tmp(m, TURN);
		if(CUDA){
			cuWrap::Gre(ret.dataPrt(), dataPrt(), tmp.dataPrt(), size());
		}else{
			std::transform(dataPrt(), dataPrt() + size(), tmp.dataPrt(), ret.dataPrt(), cpu_funcs::Gre<TYPE>);
		}
	}
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA>  MatriX<TYPE, CUDA>::operator < (const MatriX<TYPE, CUDA> & m)  const{
	if(size() != m.size()){
		Assert("矩阵大小不一致，无法进行比较");
	}
	MatriX<TYPE, CUDA> ret(*this, STRU);
	if(trans() == m.trans()){
		if(CUDA){
			cuWrap::Gre(ret.dataPrt(), dataPrt(), m.dataPrt(), size());
		}else{
			std::transform(dataPrt(), dataPrt() + size(), m.dataPrt(), ret.dataPrt(), cpu_funcs::Les<TYPE>);
		}
	}else{
		MatriX<TYPE, CUDA> tmp(m, TURN);
		if(CUDA){
			cuWrap::Gre(ret.dataPrt(), dataPrt(), tmp.dataPrt(), size());
		}else{
			std::transform(dataPrt(), dataPrt() + size(), tmp.dataPrt(), ret.dataPrt(), cpu_funcs::Les<TYPE>);
		}
	}
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA>  MatriX<TYPE, CUDA>::operator >= (const MatriX<TYPE, CUDA> & m)  const{
	if(size() != m.size()){
		Assert("矩阵大小不一致，无法进行比较");
	}
	MatriX<TYPE, CUDA> ret(*this, STRU);
	if(trans() == m.trans()){
		if(CUDA){
			cuWrap::Gre(ret.dataPrt(), dataPrt(), m.dataPrt(), size());
		}else{
			std::transform(dataPrt(), dataPrt() + size(), m.dataPrt(), ret.dataPrt(), cpu_funcs::GreEqu<TYPE>);
		}
	}else{
		MatriX<TYPE, CUDA> tmp(m, TURN);
		if(CUDA){
			cuWrap::Gre(ret.dataPrt(), dataPrt(), tmp.dataPrt(), size());
		}else{
			std::transform(dataPrt(), dataPrt() + size(), tmp.dataPrt(), ret.dataPrt(), cpu_funcs::GreEqu()<TYPE>);
		}
	}
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA>  MatriX<TYPE, CUDA>::operator <= (const MatriX<TYPE, CUDA> & m)  const{
	if(size() != m.size()){
		Assert("矩阵大小不一致，无法进行比较");
	}
	MatriX<TYPE, CUDA> ret(*this, STRU);
	if(trans() == m.trans()){
		if(CUDA){
			cuWrap::Gre(ret.dataPrt(), dataPrt(), m.dataPrt(), size());
		}else{
			std::transform(dataPrt(), dataPrt() + size(), m.dataPrt(), ret.dataPrt(), cpu_funcs::LesEqu()<TYPE>);
		}
	}else{
		MatriX<TYPE, CUDA> tmp(m, TURN);
		if(CUDA){
			cuWrap::Gre(ret.dataPrt(), dataPrt(), tmp.dataPrt(), size());
		}else{
			std::transform(dataPrt(), dataPrt() + size(), tmp.dataPrt(), ret.dataPrt(), cpu_funcs::LesEqu()<TYPE>);
		}
	}
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA>  MatriX<TYPE, CUDA>::operator == (const MatriX<TYPE, CUDA> & m)  const{
	if(size() != m.size()){
		Assert("矩阵大小不一致，无法进行比较");
	}
	MatriX<TYPE, CUDA> ret(*this, STRU);
	if(trans() == m.trans()){
		if(CUDA){
			cuWrap::Gre(ret.dataPrt(), dataPrt(), m.dataPrt(), size());
		}else{
			std::transform(dataPrt(), dataPrt() + size(), m.dataPrt(), ret.dataPrt(), cpu_funcs::Equ<TYPE>);
		}
	}else{
		MatriX<TYPE, CUDA> tmp(m, TURN);
		if(CUDA){
			cuWrap::Gre(ret.dataPrt(), dataPrt(), tmp.dataPrt(), size());
		}else{
			std::transform(dataPrt(), dataPrt() + size(), tmp.dataPrt(), ret.dataPrt(), cpu_funcs::Equ<TYPE>);
		}
	}
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE, CUDA>  MatriX<TYPE, CUDA>::operator != (const MatriX<TYPE, CUDA> & m)  const{
	if(size() != m.size()){
		Assert("矩阵大小不一致，无法进行比较");
	}
	MatriX<TYPE, CUDA> ret(*this, STRU);
	if(trans() == m.trans()){
		if(CUDA){
			cuWrap::Gre(ret.dataPrt(), dataPrt(), m.dataPrt(), size());
		}else{
			std::transform(dataPrt(), dataPrt() + size(), m.dataPrt(), ret.dataPrt(), cpu_funcs::NotEqu<TYPE>);
		}
	}else{
		MatriX<TYPE, CUDA> tmp(m, TURN);
		if(CUDA){
			cuWrap::Gre(ret.dataPrt(), dataPrt(), tmp.dataPrt(), size());
		}else{
			std::transform(dataPrt(), dataPrt() + size(), tmp.dataPrt(), ret.dataPrt(), cpu_funcs::NotEqu<TYPE>);
		}
	}
	return ret;
};