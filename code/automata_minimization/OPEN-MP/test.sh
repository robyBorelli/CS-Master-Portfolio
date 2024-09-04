params=`./4_serial | cut -d'/' -f2 | tr -d '[]' |cut -c10-`
echo "usage: $0 $params [selector]" 
echo "############## $0 $1 $2 $3 $4 $5 ################"
echo "./0_serial $1 $2 1 $4 $5 ##serial program useses 1 thread##"
(/usr/bin/time -f "%e" ./0_serial $1 $2 1 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./1_parallel_for $1 $2 $3 $4 $5 "
(/usr/bin/time -f "%e" ./1_parallel_for $1 $2 $3 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./1_serial $1 $2 1 $4 $5 ##serial program useses 1 thread##"
(/usr/bin/time -f "%e" ./1_serial $1 $2 1 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./2_parallel_task $1 $2 $3 $4 $5 "
(/usr/bin/time -f "%e" ./2_parallel_task $1 $2 $3 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./2_serial $1 $2 1 $4 $5 ##serial program useses 1 thread##"
(/usr/bin/time -f "%e" ./2_serial $1 $2 1 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./3_parallel_sections $1 $2 3 $4 $5 ##ignoring thread number. setting 3 threads##" 
(/usr/bin/time -f "%e" ./3_parallel_sections $1 $2 3 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./3_serial $1 $2 1 $4 $5 ##serial program useses 1 thread##"
(/usr/bin/time -f "%e" ./3_serial $1 $2 1 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./4_parallel_sections_splittedsort $1 $2 4 $4 $5 ##ignoring thread number. setting 4 threads##" 
(/usr/bin/time -f "%e" ./4_parallel_sections_splittedsort $1 $2 4 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./4_serial $1 $2 1 $4 $5 ##serial program useses 1 thread##"
(/usr/bin/time -f "%e" ./4_serial $1 $2 1 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./5_parallel_doublesign $1 $2 4 $4 $5 ##ignoring thread number. setting 4 threads##" 
(/usr/bin/time -f "%e" ./5_parallel_doublesign $1 $2 4 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./5_serial $1 $2 1 $4 $5 ##serial program useses 1 thread##"
(/usr/bin/time -f "%e" ./5_serial $1 $2 1 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./6_parallel_radixsort $1 $2 $3 $4 $5" 
(/usr/bin/time -f "%e" ./6_parallel_radixsort $1 $2 $3 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"

echo "./6_serial $1 $2 1 $4 $5 ##serial program useses 1 thread##"
(/usr/bin/time -f "%e" ./6_serial $1 $2 1 $4 $5 ) > f 2>&1
res=`cat f`
rm f
echo "$res"