/*
��װ glut  
GLUT3.7���ص�ַ��http://www.opengl.org/resources/libraries/glut/glutdlls37beta.zip
�������������������µ�GLUT�����µ�GLUT�汾��3.7����ѹ���� glut32.dll �� glut.dll ������  c:\windows\system32 ���棬�� glut32.lib �� glut.lib ������ VC ��װĿ¼�µ� lib Ŀ¼�£��磺\Microsoft Visual Studio 9.0\VC\lib\�£����� glut.h ������VC��װĿ¼�µ� \include\gl\ Ŀ¼�£��磺\Microsoft Visual Studio 9.0\VC\include\gl\�£�
*/
#include<gl/glut.h>
namespace Plot{
	float *X;
	float *Y;
	int num;
	float scaleX;
	float scaleY;
	float biasX;
	float biasY;
	string title = "default";
	const int high = 400;
	const int width = 400;

	void initPlot(int* argc,char** argv){
		 glutInit(argc, argv);  
		 glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
         glutInitWindowPosition(width/2, high/2);
         glutInitWindowSize(width, high);
	}
	void draw(){
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
         glLoadIdentity();
         glBegin(GL_POINTS);
		 for(int i = 0; i<num; i++){
			  glVertex3f( X[i] * scaleX + biasX, Y[i] * scaleY + biasY, 0);
		 }
         glEnd();
         glutSwapBuffers();
	}
	void plot(float * _X, float * _Y, int _num){
		X = _X;
		Y = _Y;
		num = _num;
		float Xmax = *max_element(X, X + num);
		float Xmin = *min_element(X, X + num);
		float Ymax = *max_element(Y, Y + num);
		float Ymin = *min_element(Y, Y + num);
		scaleX = float(width)/(Xmax - Xmin);
		scaleY = float(width)/(Ymax - Ymin);
		biasX = -(Xmax + Xmin)/2 * scaleX;
		biasY = -(Ymax + Ymin)/2 * scaleY;
		glutCreateWindow(title.c_str());
        glutDisplayFunc(draw);
        glutMainLoop();

	}
}