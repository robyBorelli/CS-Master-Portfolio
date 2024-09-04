params=`./4_serial | cut -d'/' -f2 | tr -d '[]' |cut -c10-`
timestamp=$(date +"%d.%H.%M")
filename="test(n$1m$2t$3l$4)date($timestamp).csv"
echo "usage: $0 $params <ITERATIONS>" 
echo "############## $0 $1 $2 $3 $4 0 $6 ################"

echo "/">temp
./test.sh 10 3 1 0 0 | grep ^./ | cut -d' ' -f1 | tr '_' ' ' | tr -d '/.' > f1
cat temp f1 > temp2
mv temp2 f1 
rm temp

for ((i = 1; i <= $6; i++)); do
	echo "TEST $i"
    ./test.sh $1 $2 $3 $4 0 | tee temp
	cat temp | grep '^[0-9][0-9]*\.' > f2
	echo "$i">temp
	cat temp f2 > temp2
	mv temp2 f2
	rm temp
	paste -d "," f1 f2 | sed 's/,/,/' > temp2
	rm f1
	mv temp2 f1
	rm f2
	cp f1 csv/$filename
	echo""
	echo""
	echo""
	echo "SHOWNING $filename"
	echo "----------"
	cat csv/$filename
	echo "----------"
done

rm f1
