/*
Reachability of patient stability
*/
Pr[<=120]([] !(doctor.stable || patient.unknown))

/*
Permanence time in non-breathing location
*/
Pr[<=120]((!patient.unknown) U (tv < 200 && t >= 10))

/*
Reachability of critical health state
*/
Pr[<=120]((!patient.unknown) U (cd_ok + ox_ok + rr_ok + tv_ok <= 1))
