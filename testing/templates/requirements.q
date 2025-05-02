/*
Non-reachability of patient stability
*/
Pr[<=1200]([] (doctor.stable imply patient.unknown))

/*
Permanence time in non-breathing location
*/
Pr[<=1200](<> (!patient.unknown && tv < 200 && t >= 10))

/*
Reachability of critical health state
*/
Pr(<> [0, 1200] ([] [0, 10] ((!patient.unknown) && (cd_ok + ox_ok + rr_ok + tv_ok <= 1))))
