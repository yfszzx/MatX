#ifndef Mat_project
#define Mat_project
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <string>
#include "Eigen/Eigen/Dense"
#include "pthread/pthread.h"
#pragma comment(lib, "pthreadVC2.lib")
using namespace Eigen;
using namespace std;
#include "MatXCuda/cuWrap.h"
#ifdef MatX_CPU_ONLY
#include "MatXCuda/emptyCuWrap.h"
#define MatX_USE_GPU false
#else
#define MatX_USE_GPU true
#endif

#include "Matrix/global.h"
#include "Matrix/matMem.h"
#include "Matrix/matCore.h"
#include "Matrix/Matrix.h"
#include "matGroup/matGroup.h"
//#include "plot/plot.h"
#include "machine/machine.h"

#include "Matrix/matMemRealise.h"
#include "Matrix/matCoreRealise.h"
#include "Matrix/assignment.h"
#include "Matrix/basicArithmetic.h"
#include "Matrix/private.h"
#include "Matrix/coeffOperator.h"
#include "Matrix/mapping.h"
#include "Matrix/others.h"
#include "Matrix/statistics.h"
#include "Matrix/logic.h"
#include "Matrix/eigen.h"
#include "matGroup/matGroupRealised.h"
#endif
