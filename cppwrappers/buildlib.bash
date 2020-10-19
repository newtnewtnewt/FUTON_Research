g++ -c -fPIC wis_speedup.cpp -o wis_speedup.o
g++ -o libwis.dll wis_speedup.o -static -Wl,--out-implib,libwis.dll.a
rm ../libwis.dll
rm ../libwis.dll.a
cp libwis.dll ..
cp libwis.dll.a ..
