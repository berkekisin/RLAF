num(1..3).
val(1..3).
1 { match(N,V) : num(N) } 1 :- val(V).
1 { match(N,V) : val(V) } 1 :- num(N).
