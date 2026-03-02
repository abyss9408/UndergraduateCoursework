#include<iostream>
#include"simd.h"
namespace{int const TST1ELTS{2}, TST2ELTS{10},TST3ELTS{100};}
void Test1()
{
    int64_t ar[TST1ELTS]={0};
    for(int i{};i<TST1ELTS;++i)
    {
        ar[i]=i+1;
    }
    
    std::cout<<"Sum:"<<sumSimd(reinterpret_cast<__m128i const*>(ar),TST1ELTS)<<'\n';

    std::cout<<"Pdt:"<<pdtSimd(reinterpret_cast<__m128i const*>(ar),TST1ELTS)<<'\n';
    
}
void Test2()
{
    int64_t ar[TST2ELTS]={0};
    for(int i{};i<TST2ELTS;++i)
    {
        ar[i]=i+1;
    }
    
    std::cout<<"Sum:"<<sumSimd(reinterpret_cast<__m128i const*>(ar),TST2ELTS)<<'\n';

    std::cout<<"Pdt:"<<pdtSimd(reinterpret_cast<__m128i const*>(ar),TST2ELTS)<<'\n';
    
}
void Test3()
{
    int64_t ar[TST3ELTS]={0};
    for(int i{};i<TST3ELTS;++i)
    {
        ar[i]=i;
    }
    
    std::cout<<"Sum:"<<sumSimd(reinterpret_cast<__m128i const*>(ar),TST3ELTS)<<'\n';

    std::cout<<"Pdt:"<<pdtSimd(reinterpret_cast<__m128i const*>(ar),TST3ELTS)<<'\n';
    
}

int main()
{
    int i{};
    std::cin>>i;
    switch(i)
    {
    case 1:
        Test1();
        break;
    case 2:
        Test2();
        break;
    case 3:
        Test3();
        break;
    }

}