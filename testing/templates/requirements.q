/*
Non-reachability of patient stability
*/
Pr[<=120]([] (doctor.stable imply patient.unknown))

/*
Permanence time in non-breathing location
*/
Pr[<=120](<> (!patient.unknown && tv < 200 && t >= 10))

/*
Reachability of critical health state
*/
Pr(<> [0, 120] ([] [0, 10] ((!patient.unknown) && (cd_ok + ox_ok + rr_ok + tv_ok <= 1))))
