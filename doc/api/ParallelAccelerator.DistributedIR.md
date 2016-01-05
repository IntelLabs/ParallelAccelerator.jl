# ParallelAccelerator.DistributedIR

## Internal

---

<a id="method__checkparforsfordistribution.1" class="lexicon_definition"></a>
#### checkParforsForDistribution(state::ParallelAccelerator.DistributedIR.DistIrState) [¶](#method__checkparforsfordistribution.1)
All arrays of a parfor should distributable for it to be distributable.
If an array is used in any sequential parfor, it is not distributable.


*source:*
[ParallelAccelerator/src/distributed-ir.jl:233](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/distributed-ir.jl)

---

<a id="method__get_arr_dist_info.1" class="lexicon_definition"></a>
#### get_arr_dist_info(node::Expr,  state,  top_level_number,  is_top_level,  read) [¶](#method__get_arr_dist_info.1)
mark sequential arrays


*source:*
[ParallelAccelerator/src/distributed-ir.jl:161](file:///home/etotoni/.julia/v0.4/ParallelAccelerator/src/distributed-ir.jl)

