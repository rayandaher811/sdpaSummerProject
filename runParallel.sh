# ./solveSdp.exe <bStart> <bEnd> <bJumps> &
# This command runs the problem we have from bStart to bEnd withbJumps

start_time=$(date +%s.%6N)

# Parallel run over 4 threads
./solveSdp.exe 0 25 0.1 &
./solveSdp.exe 25 50 0.1 &
./solveSdp.exe 50 75 0.1 &
./solveSdp.exe 75 100 0.1 &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

end_time=$(date +%s.%6N)

parallelTime=$(echo "scale=6; $end_time - $start_time" | bc)

start_time=$(date +%s.%6N)

# Single thread run
./solveSdp.exe 0 100 0.1 &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

end_time=$(date +%s.%6N)
singleThreadTime=$(echo "scale=6; $end_time - $start_time" | bc)

echo 'Single thread time is ', $singleThreadTime
echo 'multithread time is ', $parallelTime
