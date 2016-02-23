
class array_operate{
public:
	int dimen;
	void add(float *arr,float *arr1,float *arr2,float x);
	void simple_add(float *arr1,float *arr2,float x);
	float dot(float *arr1,float *arr2);
	void zero(float *arr);
	void scale(float *arr,float x);
	void clone(float *arr_from,float *arr_to);
	float length(float *arr);
};
struct search_set{
	float accept_scale;
	float wp_value;
	float wp_deriv;
	bool strict;	
	int max_round;
	float mg_param;
	int cg_reset_num;
	int L_save_num;
	bool debug;		
	char mod;
	float init_step;
	float step;
	float r_step;
	int dimen;
	float drv_scl;
	float max_search_angle;
	float max_step;
	char file_name[50];
	void record(float stp);
	int memery(bool show=true);
	search_set();
	void show();
	void set();
	void reset_step();
	void save(string path);
	bool read(string path);
};
struct LG_STRUCT{
	float *s_data;	
	float *y_data;
	float *alf;
	float *ro;
	float **s;
	float **y;
	float save_num;
	int dimen;
	LG_STRUCT();
	~LG_STRUCT();
	void malloc(int dmn,int m);
	void free_mem();
};
class search_tool:public array_operate{
	//调用虚函数cacal()以计算pos处的值和梯度，结果分别保留在result和deriv中

public:
	bool pause_flag;
	float current_step;
	float init_deriv;
	search_tool(string path);
	~search_tool();
	void search_init(int dmn,float *rlt_p,float *pos_p,float *drv_p);

	void set_search();
	void reset_step();
	void save_search();
	bool search(float r);
	
protected:
	virtual bool show_and_control(int)=0;
	virtual void cacul()=0;
	search_set set;
	LG_STRUCT Ld;
private:
	float *result;
	float *pos;
	float *deriv;	
	string root;
	
	float *pos_init;
	float *deriv_tmp;	
	float *direct;
	float *tmp_array;
	float deriv_len;
	void free_mem();
	bool pause();
	float interpolation(float x,float v0,float v1,float derv0,float derv1);
	float wolfe_powell(float step);
	bool momentum_grad(int rounds);
	bool move();
	bool fast_grad(int rounds);
	bool conj_grad(int rounds);
	

	bool LBFGS(int rounds);

};

