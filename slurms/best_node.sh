#!/bin/bash
###################################### get_best_node <cpus> <mem_GB> [partitions]
get_best_node() {
    local c="${1:-1}" m="${2:-1}" p="${3:-}"
    sinfo -N -h ${p:+-p "$p"} -O "NodeList:40,Partition:24,StateCompact:12,CPUsState:24,Memory:14,AllocMem:14" | awk -F'[ /]+' -v c="$c" -v m="$m" '
    BEGIN{max=-1}
    $1~/^(dgx|gpu)/||toupper($2)~/^(DGX|GPU)/||$3~/[-*~#!%$@^]$/||seen[$1]++{next}
    $3!~/^(idle|mix|mixed)$/{next}
    {p=$2; sub(/\*$/,"",p); idle=$5+0; tc=$7+0; mem=$8-$9; if(mem<0)mem=0
     if(idle<c||mem<m*1024)next
     s=idle/tc+mem/$8
     if(s>max){max=s;n=$1;bp=p;bm=mem;bi=idle}}
    END{if(n){printf "[BASH-INFO] %s (%s) | Unalloc Mem: %.1f GB | Idle CPUs: %d\n",n,bp,bm/1024,bi > "/dev/stderr"; print n, bp}
        else print "[BASH-INFO] No node fits the request." > "/dev/stderr"}'
}
