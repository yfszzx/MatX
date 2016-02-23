/*����˵��:
��1��VS����:property ->cuda c/c++ ->Device ->code generation ->sm_**��Ҫ��sm_30���ϣ����罫ԭ����sm_10��Ϊsm_30�����)
��2��VS����:property ->linker ->input ->addition dependencies ->���� curand.lib;cublas.lib;
��3) ���ļ�pthreadVC2.dll������.exe�ļ���ͬһĿ¼��
(4)POSIX Threads for Win32����Ŀ��ר��Ϊwin32������һ��pthread��li���ص���exe��ѹ֮�󣬻�õ�����Ŀ¼��
���У�Pre-built.2�����Ѿ�����õ�lib�Լ�dll��ͬʱ������һЩ��Ҫ��ͷ�ļ��������е�include�ļ��к�lib�ļ���copy��VC�İ�װĿ¼��
���磬VC6.0�Ļ�����Ĭ�ϰ�װ������Ҫcopy����C:\Program Files\Microsoft Visual Studio\VC98

���ţ��ڱ�̵�ʱ������pthreadVC2.lib���ɣ�

   1: #pragma comment(lib, "pthreadVC2.lib")
 �ο���http://www.cnblogs.com/ayanmw/archive/2012/08/06/2625275.html
*/

#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <regex>
#include <sys\stat.h> 
#include<cmath>
#include<cstdlib>
#include<cstring>
#include<cfloat>
#include<ctime>
#include <direct.h> 
#include <sstream>
#include <io.h>
#include <iomanip>  
#include <stdlib.h>
#include <cmath>  
#include <cfloat> 
#include <algorithm>
#include <conio.h>   //��������
#include <WINSOCK2.H>
#pragma comment(lib,"ws2_32.lib")
using namespace std;
#define coutd if(cout_show)cout<<'\n'
//#define coutd cout<<"\n"
#define memzero(x) memset((x),0,_msize((x)))
#define safe_free(x) if((x)!=NULL)delete [] (x)
#define safe_gpu_free(x)  if((x)!=NULL)cudaFree((x))
#define Mrand (rand()%(RAND_MAX+1)*(RAND_MAX+1)+rand())
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h> 
#include <thrust/fill.h>
#include <thrust/replace.h> 
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <curand.h>

using namespace std;
bool cout_show=true;

int g_threads=-1;
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define CUBT(x) show_cublas_error((x),__LINE__,__FILE__);
#define CUDA_CHECK show_cuda_error(__LINE__,__FILE__);
#define CHECK_CURAND(err) do{\
    if( (err) != CURAND_STATUS_SUCCESS ){\
        fprintf(stderr, "CURAND error %d at file %s line %d.\n", (int)(err), __FILE__, __LINE__);\
  getchar();\
    }}while(0);
#define CUDA_ERR_FLAG flag_cuda_error(__LINE__,__FILE__)



bool flag_cuda_error(int line,char *file){ 
	cudaThreadSynchronize();
    cudaError_t err=cudaGetLastError();//������һ�δ���Ĵ������
   if(err)
	{
		coutd<<"CUDA "<<cudaGetErrorString(err)<<" at line "<<line-1<<"  file "<<file<<endl;;//��ʾ��������
		getchar();
		return true;
	}
   return false;
}


void g_gpu_init(){
	if(g_threads!=-1)return;
	 int device_count;
	 if( cudaGetDeviceCount(&device_count) ){
		 coutd<<"������û�з��ֿ��õ��Կ��豸";
		 getchar();
		 getchar();
		 return;
	 }
	coutd<<"����"<<device_count<<"�����õ��Կ�:";
	for(int i=0;i<device_count;i++){
		 struct cudaDeviceProp device_prop;
		 if(cudaGetDeviceProperties(&device_prop,i)==cudaSuccess){
			 coutd<<"\t"<<i<<"."<<device_prop.name;
		 }
	}
	int idx=0;
/*	if(device_count>1){
		do{
		coutd<<"��ѡ���Կ���������ţ�";
		cin>>idx;
		}while(idx<0||idx>=device_count);
	}*/
  	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,idx);
	CUDA_CHECK;
	g_threads=prop.maxThreadsPerBlock;
	coutd<<"threads per block:"<<g_threads;
	coutd<<"��ʼ��CUBLAS";
	CUBT(cublasCreate(&cublasHandle));
}
void show_gpu_data(float *data,int num,string illu=""){//����ʹ��
	float *show=new float[num];
	cudaMemcpy(show,data,sizeof(float)*num,cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	coutd<<illu;
	for(int i=0;i<num;i++)cout<<" "<<show[i];
	
}
float gpu_read_unit_value(float *from){
		float ret;
		cudaMemcpy(&ret,from,sizeof(float),cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		return ret;
	}
void gpu_write_unit_value(float *from,float value){
		cudaMemcpy(from,&value,sizeof(float),cudaMemcpyHostToDevice);
		CUDA_CHECK;
	}
void show_cpu_data(float *data,int num=100,string illu=""){//����ʹ��
	coutd<<illu;
	for(int i=0;i<num;i++)cout<<" "<<data[i];

}

void show_memery_size(int m,char mod='g',string illu=""){
	if(cout_show){
		cout<<illu<<float(m)/1048576<<"M";
		if(mod=='g')cout<<"�Դ�";
		if(mod=='c')cout<<"�ڴ�";
	}
}	
struct{
	bool check_folder(string p){
		struct _stat fl;
		string path=p;
		int i=path.size();
		if(path[i-1]=='\\')path[i-1]=0;
		if (!((_stat(path.c_str(), &fl) == 0) && (fl.st_mode & _S_IFDIR)))return 0;
		return 1;
	}
	bool check_file(string p){
		ifstream in(p);  
		if(!in)	return 0;
		in.close();
		return 1;
	}
	void create_folder(string path){
		if(!check_folder(path))mkdir(path.c_str());
	}
	
	bool copy(string ffrom,string fto){
		char ch;
		ifstream in(ffrom,ios::in|ios::binary);             //�������ļ�
		ofstream out(fto,ios::out|ios::binary);              //������ļ�
		if(!in){
			cout<<"can not open the file \n"<<ffrom<<endl;
			return 0;
		}
		if(!out){
			cout<<"can not open the file \n"<<fto<<endl;
			return 0;
		}
		while (in.get(ch))//ʵ�ָ��ƹ���
			out.put(ch);          

		in.close();
		out.close();
		return 1;
	}
	void show_project_name(string path,string symbol_file_name){
		_finddata_t FileInfo;
		string search =path+"*";
		long Handle = _findfirst(search.c_str(), &FileInfo); 
		if (Handle == -1L){
			coutd<<"Ĭ��Ŀ¼"<<search<<"������\n";
			return ;
		}    
		do{
			if (FileInfo.attrib &_A_SUBDIR){
				if( (strcmp(FileInfo.name,".") != 0 ) && (strcmp(FileInfo.name,"..") != 0)){
						string p=path+FileInfo.name+"\\"+  symbol_file_name;
						if(!check_file(p))continue;
						coutd<<"\t<"<<FileInfo.name<<"> ";
				}
			}				
		}while (_findnext(Handle, &FileInfo) == 0); 
		_findclose(Handle);
	}


}file_opt;


template <typename A,typename B>
struct g_array_type_trans {  
__device__  B operator ()( const A & x) const {  
return x;  
}  
}; 
template <typename A,typename B>
void array_type_trans(A *from,B *to,int dimen){//����ת��
	thrust :: device_ptr <A> f(from); 
	thrust :: device_ptr <B> t(to); 
	thrust :: transform (f, f+dimen ,t , g_array_type_trans<A,B>());
}

__global__ void g_array_add_to_matrix(float *w,float *o,float param,int dimen,int data_num){
	int node_idx=blockIdx.x;
	int data_idx=blockIdx.y*blockDim.x+threadIdx.x;
	__shared__ float t;
	if(threadIdx.x==0)t=w[node_idx];
	__syncthreads();
	if(data_idx<data_num){
		o[data_idx*dimen+node_idx]+=t*param;
	}
	
}
void array_add_to_matrix(float *mtx,float *ary,float param,int dimen,int num){//�ھ�����ÿ�м�һ������
	int block_y=(num+g_threads-1)/g_threads;
	dim3 blk(dimen,block_y);
	g_array_add_to_matrix<<<blk,g_threads>>>(ary,mtx,param,dimen,num);		
	CUDA_CHECK;
}
struct g_array_float_plus_double{  
__device__ double operator ()( float & x,double &y) const {  
return x+y;  
}  
}; 
void array_float_plus_double(float *arr,double *dbl,int dimen){//float ��double ��ӣ����������double��
		thrust::device_ptr<float> a ( arr );
		thrust::device_ptr<double> b (dbl);
		thrust:: transform (a , a+dimen ,b,b,g_array_float_plus_double());
}
float	array_sum(float *arr,int dimen){//ʸ������֮��
		thrust::device_ptr<float> a ( arr );
		return  thrust::reduce (a , a+dimen, (float) 0, thrust::plus <float>());
}
float array_length(float *arr,int dimen){
	float ret;
	CUBT(cublasSnrm2 (cublasHandle,dimen, arr, 1,&ret));
	return ret;
}