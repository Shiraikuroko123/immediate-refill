# Exact worst-case ratio of immediate refill

Fix \(P\) and positive prices \(c_1,\ldots,c_P\) for a common unit of work at each active width, with \(c_0=0\). Consider the equal-vote, ordered, nonpreemptive model with zero extra starts. Take the supremum over all panel sizes, positive path lengths, votes and absolute quotas. With \(\widehat c\) the least concave majorant and \(G\) immediate indexed refill,
\[
\sup_{\text{panels}}\frac{T_G(c)}{T_{\rm opt}(c)}
=\max_{1\le j\le P}\frac{\widehat c_j}{c_j}=\gamma .
\]
The benchmark chooses a pathwise best schedule that obeys the same gate at every observed history. On a particular panel the ratio can be smaller. The upper bound follows from greedy optimality on concave clocks and pointwise comparison of the two prices. [Upper bound and hardware calculations](README.md) · [Majorization theorem](../width-generalization/README.md)

**A matching family.** Suppose \(\gamma>1\) and take a maximizing \(j\). The point \((j,\widehat c_j)\) lies on a chord between integer endpoints \(a<j<b\) of the upper hull, with \(\widehat c_a=c_a\) and \(\widehat c_b=c_b\). Let \(S=\operatorname{lcm}(b-a,j-a)\) and \(L=S/(j-a)\). The final tasks consist of \(a\) paths of work \(L\), followed by \(S\) paths of work one. Choose an integer \(d>S+a+P\) divisible by \(P\) and prepend \(d\) paths of work \(\epsilon>0\). The prefix votes are A. All the next \(a+S\) paths vote B. Take \(N=2(d+b-1)\) and \(Q=d+a+S\). Any remaining votes can be assigned B.

Both schedules fill all \(P\) slots during the prefix. With \(s\) returned A votes and no others, the vote bound on active width is \(\min(N-s,d+b-s)\). It permits \(P\) paths at the start of every prefix batch since \(s\le d-P\). The prefix waves finish simultaneously and cost the same \(D_\epsilon=(d/P)c_P\epsilon\). After all \(d\) prefix votes return, \(R-g=2b-2\), where \(R\) is the number of unresolved paths and \(g\) is the plurality margin. At every subsequent B completion the leader remains A and \(R\) and \(g\) each decrease by one. Therefore the active-width allowance from the vote bound remains \(b\), up to the remaining quota. There is no strict lock, and the panel returns upon completion of all \(Q\) jobs.

Greedy fills to width \(b\). Its \(b-a\) short-job slots serve \(S\) jobs in an integer number of full waves, totaling \(D_b=S/(b-a)\) virtual steps. The long jobs occupy \(a\) slots until virtual time \(L\) after the prefix. A comparator limits the suffix to \(j\) slots. Its \(j-a\) short-job slots finish together in \(L\) full waves, precisely when all long paths end. At \(a=0\) greedy terminates at \(D_b\), and \(c_0=0\) accounts for the remainder. The greedy and comparator suffix costs are
\[
D_b c_b+(L-D_b)c_a = L\widehat c_j,\qquad Lc_j .
\]
Consequently \(T_G/T_{\rm opt}\) is at least \((D_\epsilon+L\widehat c_j)/(D_\epsilon+Lc_j)\). Letting \(\epsilon\) decrease to zero proves the lower bound on the supremum. For \(\gamma=1\), \(T_{\rm opt}\le T_G\) suffices. The same limiting argument permits positive integer workloads by taking unit work on the prefix and scaling every suffix workload by the same unbounded integer.

The simulator independently applies the indexed gate at each event. It recovers the two claimed suffix occupancy profiles on all 210 triples \(0\le a<j<b\le P\) for \(P=2,\ldots,8\). Run `python -B tracks/interference/clock-uncertainty/sharpness.py` from the project root. [Simulation](sharpness.py) · [Counts](results/sharpness.json)
