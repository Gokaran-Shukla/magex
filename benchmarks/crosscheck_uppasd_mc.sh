for T in 20 28 32 36 40; do
  mkdir -p T$T; cp posfile momfile jfile kfile T$T/ 2>/dev/null
  sed -e "s/^ncell .*/ncell 13 13 13/" -e "s/^10000 300 1.0e-16 0.01/3000 $T/" -e "s/^temp 300/temp $T/" -e "s/^mcNstep 20000/mcNstep 6000/" inpsd.dat.template > T$T/inpsd.dat
  printf "do_avrg Y\navrg_step 10\n" >> T$T/inpsd.dat
  (cd T$T && s=$(date +%s.%N) && ${UPPASD:-uppasd} > out.log 2>&1; e=$(date +%s.%N); echo "$e - $s" | bc > time.log)
done
