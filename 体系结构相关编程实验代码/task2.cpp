#include<iostream>
#include<cmath>
#include <windows.h>
#include <stdlib.h>
using namespace std;
const int n = pow(2,8);
const int Count = 200000;
int a[n];
void init()
{
    for(int i = 0; i<n; i++)
    {
        a[i] = i;
    }
}
int sum = 0;
int sum2 = 0;
int main()
{
    long long head, tail, freq; // timers
    double total_time = 0.0;
    init();

    /*
    //平凡算法
    for(int k = 0; k<Count; k++)
    {

        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        //start time
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int i = 0; i< n; i++)
        {
            sum += a[i];
        }
        //end time
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        total_time += (tail - head) * 1000.0 / freq;

    }
    */

    //循环展开算法
    for(int k = 0; k<Count; k++)
    {

        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        //start time
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int i = 0; i< n;i+=2)
        {
            sum += a[i];
            sum2 += a[i+1];
        }
        sum += sum2;
        //end time
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        total_time += (tail - head) * 1000.0 / freq;

    }

    cout<<"N:" << n<< endl;
    cout<<"Count:" << Count<< endl;
    cout<< total_time <<endl;
    cout<<total_time / Count;
    return 0;
}
