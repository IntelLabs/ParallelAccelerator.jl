# ParallelAccelerator

## Internal

---

<a id="method____init__.1" class="lexicon_definition"></a>
#### __init__() [¶](#method____init__.1)
Called when the package is loaded to do initialization.


*source:*
[ParallelAccelerator/src/ParallelAccelerator.jl:209](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/ParallelAccelerator.jl#L209)

---

<a id="method__embed.1" class="lexicon_definition"></a>
#### embed() [¶](#method__embed.1)
This version of embed tries to use JULIA_HOME to find the root of the source distribution.
It then calls the version above specifying the path.


*source:*
[ParallelAccelerator/src/ParallelAccelerator.jl:185](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/ParallelAccelerator.jl#L185)

---

<a id="method__embed.2" class="lexicon_definition"></a>
#### embed(julia_root) [¶](#method__embed.2)
Call this function if you want to embed binary-code of ParallelAccelerator into your Julia build to speed-up @acc compilation time.
It will attempt to add a userimg.jl file to your Julia distribution and then re-build Julia.


*source:*
[ParallelAccelerator/src/ParallelAccelerator.jl:138](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/ParallelAccelerator.jl#L138)

---

<a id="method__getpackageroot.1" class="lexicon_definition"></a>
#### getPackageRoot() [¶](#method__getpackageroot.1)
Generate a file path to the directory above the one containing this source file.
This should be the root of the package.


*source:*
[ParallelAccelerator/src/ParallelAccelerator.jl:126](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/ParallelAccelerator.jl#L126)

---

<a id="method__getpsemode.1" class="lexicon_definition"></a>
#### getPseMode() [¶](#method__getpsemode.1)
Return internal mode number by looking up environment variable "PROSPECT_MODE".


*source:*
[ParallelAccelerator/src/ParallelAccelerator.jl:57](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/ParallelAccelerator.jl#L57)

---

<a id="method__gettaskmode.1" class="lexicon_definition"></a>
#### getTaskMode() [¶](#method__gettaskmode.1)
Return internal mode number by looking up environment variable "PROSPECT_TASK_MODE".
If not specified, it defaults to NO_TASK_MODE, or DYNAMIC_TASK_MODE when 
getPseMode() is TASK_MODE.


*source:*
[ParallelAccelerator/src/ParallelAccelerator.jl:100](https://github.com/IntelLabs/ParallelAccelerator.jl/tree/44944f13cdcd8839ae646ee3ca66dbafdec20db5/src/ParallelAccelerator.jl#L100)

