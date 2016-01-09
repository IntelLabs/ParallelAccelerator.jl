# ParallelAccelerator.DistributedIR

## Internal

---

<a id="method__checkparforsfordistribution.1" class="lexicon_definition"></a>
#### checkParforsForDistribution(state::ParallelAccelerator.DistributedIR.DistIrState) [¶](#method__checkparforsfordistribution.1)
All arrays of a parfor should distributable for it to be distributable.
If an array is used in any sequential parfor, it is not distributable.


*source:*
[ParallelAccelerator/src/distributed-ir.jl:232](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/distributed-ir.jl#L232)

---

<a id="method__get_arr_dist_info.1" class="lexicon_definition"></a>
#### get_arr_dist_info(node::Expr,  state,  top_level_number,  is_top_level,  read) [¶](#method__get_arr_dist_info.1)
mark sequential arrays


*source:*
[ParallelAccelerator/src/distributed-ir.jl:162](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/distributed-ir.jl#L162)

