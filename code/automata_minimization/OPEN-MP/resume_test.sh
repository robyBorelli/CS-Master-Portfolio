echo "usage: $0 <test(n[0-9]+m[0-9]+t[0-9]+l[0-9]+)*.csv> <ADDITIONAL_ITERATIONS>" 
filename=$1
it=$2

first=`cat $filename | head -n 1 | rev | cut -d',' -f1 | rev | tr -d '\r\n\t '`
last=$((first + it))
((first++))
i=$first

[[ $filename =~ n([0-9]+)m([0-9]+)t([0-9]+)l([0-9]+) ]] && \
n_value="${BASH_REMATCH[1]}" \
m_value="${BASH_REMATCH[2]}" \
t_value="${BASH_REMATCH[3]}" \
l_value="${BASH_REMATCH[4]}"

echo "SHOWNING $filename"
echo "----------"
cat $filename
echo "----------"

echo "RESUMING ############## ./multi_test.sh $n_value $m_value $t_value $l_value 0 $first ################"

cat $filename > f1

for ((; i <= $last; i++)); do
	echo "TEST $i"
    ./test.sh $n_value $m_value $t_value $l_value 0 | tee temp
	cat temp | grep '^[0-9][0-9]*\.' > f2
	echo "$i">temp
	cat temp f2 > temp2
	mv temp2 f2
	rm temp
	paste -d "," f1 f2 | sed 's/,/,/' > temp2
	rm f1
	mv temp2 f1
	rm f2
	cp f1 $filename
	echo""
	echo""
	echo""
	echo "SHOWNING $filename"
	echo "----------"
	cat $filename
	echo "----------"
done

rm f1
