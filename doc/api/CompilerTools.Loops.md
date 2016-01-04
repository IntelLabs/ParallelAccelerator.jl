# CompilerTools.Loops

## Exported

---

<a id="type__domloops.1" class="lexicon_definition"></a>
#### CompilerTools.Loops.DomLoops [¶](#type__domloops.1)
A type that holds information about which basic blocks dominate which other blocks.
It also contains an array "loops" of all the loops discovered within the function.
The same basic block may occur as a member in multiple loop entries if those loops are nested.
This module doesn't currently help identify these nesting levels but by looking at loop members it is easy enough to figure out.


*source:*
[CompilerTools/src/loops.jl:59](file:///home/etotoni/.julia/v0.4/CompilerTools/src/loops.jl)

---

<a id="type__loop.1" class="lexicon_definition"></a>
#### CompilerTools.Loops.Loop [¶](#type__loop.1)
A type to hold information about a loop.
A loop has a "head" that dominates all the other blocks in the loop.
A loop has a back_edge which is a block that has "head" as one of its successors.
It also contains "members" which is a set of basic block labels of all the basic blocks that are a part of this loop.


*source:*
[CompilerTools/src/loops.jl:43](file:///home/etotoni/.julia/v0.4/CompilerTools/src/loops.jl)

## Internal

---

<a id="method__compute_dom_loops.1" class="lexicon_definition"></a>
#### compute_dom_loops(bl::CompilerTools.CFGs.CFG) [¶](#method__compute_dom_loops.1)
Find the loops in a CFGs.CFG in "bl".


*source:*
[CompilerTools/src/loops.jl:211](file:///home/etotoni/.julia/v0.4/CompilerTools/src/loops.jl)

---

<a id="method__findloopinvariants.1" class="lexicon_definition"></a>
#### findLoopInvariants(l::CompilerTools.Loops.Loop,  udinfo::Dict{CompilerTools.LivenessAnalysis.BasicBlock, CompilerTools.UDChains.UDInfo},  bl::CompilerTools.LivenessAnalysis.BlockLiveness) [¶](#method__findloopinvariants.1)
Finds those computations within a loop that are iteration invariant.
Takes as input:
   l - the Loop to find invariants in
   udinfo - UDChains (use-definition chains) for the basic blocks of the function.
   bl - LivenessAnalysis.BlockLiveness with liveness information for variables within the function.


*source:*
[CompilerTools/src/loops.jl:86](file:///home/etotoni/.julia/v0.4/CompilerTools/src/loops.jl)

---

<a id="method__findloopmembers.1" class="lexicon_definition"></a>
#### findLoopMembers(head,  back_edge,  bbs) [¶](#method__findloopmembers.1)
Find all the members of the loop as specified by the "head" basic block and the "back_edge" basic block.
Also takes the dictionary of labels to basic blocks.
Start with just the head of the loop as a member and then starts recursing with the back_edge using flm_internal.


*source:*
[CompilerTools/src/loops.jl:205](file:///home/etotoni/.julia/v0.4/CompilerTools/src/loops.jl)

---

<a id="method__flm_internal.1" class="lexicon_definition"></a>
#### flm_internal(cur_bb,  members,  bbs) [¶](#method__flm_internal.1)
Add to the "members" of the loop being accumulated given "cur_bb" which is known to be a member of the loop.


*source:*
[CompilerTools/src/loops.jl:183](file:///home/etotoni/.julia/v0.4/CompilerTools/src/loops.jl)

---

<a id="method__isinloop.1" class="lexicon_definition"></a>
#### isInLoop(dl::CompilerTools.Loops.DomLoops,  bb::Int64) [¶](#method__isinloop.1)
Takes a DomLoops object containing loop information about the function.
Returns true if the given basic block label "bb" is in some loop in the function.


*source:*
[CompilerTools/src/loops.jl:66](file:///home/etotoni/.julia/v0.4/CompilerTools/src/loops.jl)

