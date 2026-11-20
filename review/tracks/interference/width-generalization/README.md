# When immediate refill is optimal for any number of slots

Let \(P\ge1\) be the number of slots, and write \(c_0=0\) and \(c_k>0\) for the time to advance \(k\) active paths by one common unit of work. In the equal-vote, zero-serial-overrun model, immediate greedy refill minimizes decision time on **every** panel if and only if
\[
2c_j\ \ge\ c_{j-1}+c_{j+1},\qquad j=1,\dots,P-1.
\]
The cost curve may decrease at some widths. The universal quantifier ranges over path lengths, votes, panel sizes and absolute quotas on a server with \(P\) slots. It permits foreknowledge by a comparator, provided its dispatches satisfy the information-based admission contract. For one slot the family of inequalities is empty.

The assumptions are a fixed launch order, strictly positive fixed path work, equal weights, a terminal vote only, immediate return on strict certification or full quota exhaustion, nonpreemptive service until return and equal progress across every active path in the virtual clock. A change of clock \(d\tau=dt/c_k\) makes the work rate of each active path one. An interval of empty-system idling can be dropped from a comparator without changing its information, work or feasible dispatches.

## Two properties of greedy on different work vectors

With zero extra starts, every feasible complete schedule runs the same indexed jobs \(1,\ldots,J\) to completion, where \(J=\min(Q,K)\) and \(K\) is the decisive serial prefix (or \(N\) on exhaustion). Thus \(J\) depends on the fixed votes and quota, and all schedules on the same panel have the same total work. The indexed cumulative safe-start boundary \(F(S)\), incorporating quota, slot access and the relative budget, is nondecreasing when the set \(S\) of indexed observed votes grows. These properties are proved in the existing zero-allowance results. [Zero-allowance proof](../affine/README.md) · [Indexed/count identity at reachable histories](../../../manuscript/journal-revision/appendices.tex)

If the work requirements \(w'_i\le w_i\) are shortened while retaining the votes, greedy on \(w'\) starts and completes every job no later than any admissible schedule on \(w\), measured in virtual time. To prove this, consider in order the launches of the schedule on \(w\). At its launch of job \(j\), all its completed jobs had earlier indices and reached their launch no later on the shortened greedy schedule by induction. With shorter service, they also completed no later. Greedy therefore has enough observations to reach that cumulative launch count by \(F\)'s monotonicity. Positive work excludes a newly launched job from the simultaneous completions motivating that launch. If shortened greedy has already returned, it already started the same \(J\) jobs. This also proves the statement when \(w'=w\).

Let \(h_{G,w}(t)\) be the busy width under greedy in virtual time, set to zero after termination. On every nonterminal reachable history with \(s=|S|\), \(R=N-s\), and plurality margin \(g\), this width, after greedy refill, is
\[
g_*(S)=\min\!\left(P,Q-s,R,\ \max\!\left(0,\left\lfloor (R-g)/2\right\rfloor+1\right)\right).
\]
The same expression is zero at termination. Adding a completed vote reduces \(R\) by one, and can lower \(g\) by at most one. Each term in this minimum is therefore nonincreasing as observations grow. The cross-work induction just proved shows that at the same virtual time the shortened greedy execution includes every completion of the original greedy execution. Hence
\[
h_{G,w'}(t)\ \le\ h_{G,w}(t).
\]
This is the property that permits comparison at every intermediate width. In particular, the full width profile of a single greedy run is nonincreasing.

## Compressing a competitor's timeline

Consider a finite admissible schedule \(A\) of duration \(D\), and any union of intervals \(E\subseteq[0,D]\). Its width is \(h_A\). For \(0<\epsilon<1\), change its virtual time using
\[
\phi_\epsilon(t)=\int_0^t [\mathbf 1_E(u)+\epsilon\mathbf 1_{E^c}(u)]\,du .
\]
Keep all starts, completions and decisions in their original order at transformed times. Give each completed job of \(A\) the transformed work \(w'_i=\phi_\epsilon(e_i)-\phi_\epsilon(s_i)\), where \(s_i,e_i\) are its virtual start and finish. Leave the work of the unlaunched paths unchanged. The transformed work is positive and at most \(w_i\). The transformed schedule \(A'\) is feasible on that shorter panel. Every path is active throughout its transformed interval and progresses at unit virtual rate, while the strictly increasing map preserves event order, simultaneous completions, active slot counts and the observations available at every decision.

Set \(D_\epsilon=\phi_\epsilon(D)\). Greedy on \(w'\) completes by \(D_\epsilon\), by the equal-work-vector case of the launch induction. Both schedules run the same jobs, so all the transformed work satisfies
\[
\int_E h_A+\epsilon\int_{[0,D]\setminus E}h_A
  =\sum_{i\le J}w'_i
  =\int_0^{D_\epsilon}h_{G,w'}(u)\,du
  \ \le\ \int_0^{D_\epsilon} h_{G,w}(u)\,du.
\]
The final integral uses the original greedy profile, which is bounded by \(P\) and has no dependence on \(\epsilon\). Taking \(\epsilon\downarrow0\) gives
\[
                         \int_E h_A(u)\,du\ \le\ \int_0^{|E|}h_{G,w}(u)\,du .
\]
All schedules have the same total work. Together with the nonincreasing greedy profile this shows that greedy's busy-width profile majorizes the comparator's profile rearranged in decreasing order (extend both profiles by zero to a common domain).

## Wall time and necessity

For an integer \(1\le j<P\), choose \(E=\{t:h_A(t)>j\}\). Subtracting \(j|E|\) gives
\[
\int (h_A-j)_+ \ \le\ \int_0^{|E|}(h_{G,w}-j)\ \le\ \int (h_{G,w}-j)_+ .
\]
Write \(\theta_j=2c_j-c_{j-1}-c_{j+1}\). The discrete identity
\[
                        c_k=c_1k-\sum_{j=1}^{P-1}\theta_j(k-j)_+
\]
holds for all \(0\le k\le P\). Equal total work cancels the first term in a comparison. Nonnegative \(\theta_j\) and the threshold inequalities yield \(T_G\le T_A\).

For necessity, fix any violated index \(j\) and take a panel with even \(N\ge\max(2j,P)\), quota \(Q=j+1\), and all votes for the same label. No subset of \(j\) votes can certify, so all \(Q\) launches are safe at the outset. Give the first \(j-1\) paths work two and the next two work one. Greedy takes \(c_{j+1}+c_{j-1}\), while starting \(j\) paths and replacing the short one at its finish takes \(2c_j\). For \(j=1\), \(c_0=0\) accounts for greedy completing immediately after the first step. The schedules return on certification or full quota exhaustion. This recovers necessity for every fixed physical slot count. [Earlier witness family](../affine/README.md)

## Independent finite check and scope

The integer-time oracle evaluates every feasible admission count at each virtual tick. It checks the maximum value of each individual convex threshold feature against greedy on the same panel, using dynamic programming over remaining job work and a direct scan of all candidate serial prefixes. The inputs include an exhaustive four-slot, quota-four family and seeded four- through eight-slot families allowing replenishment. All 4,488 nontrivial threshold comparisons passed across 1,416 panel configurations, covering 1,651,840 complete integer-time histories. These counts sum over configurations and include schedules that yield the same occupancy. The continuous-time comparison above covers the intervening launch times. [Oracle](check.py) · [Results](results.json)

Run `python -B tracks/interference/width-generalization/check.py` from the project root. To save a fresh report, add `--out` and an unused filename.

This derivation uses a classical convex-order criterion for scalar width profiles. The technical question for attribution is whether an existing scheduling result already proves profile majorization for ordered nonpreemptive dispatch with an observation-dependent, shrinking active width. The earlier literature audit covers subset-capacity inequalities and three-slot dominance. A separate search is needed for this stronger statement. [Earlier literature comparison](../novelty/README.md)
