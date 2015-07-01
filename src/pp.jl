function recursive_print(e::Expr,prefix)
   println(prefix,"Expr ",e.head)
   subprefix=string("    ",prefix)
   for i in e.args
        recursive_print(i,subprefix)
   end
end

function recursive_print(s::Symbol,prefix)
    println(prefix,":", s)
end

function recursive_print(node::Any,prefix)
    println(prefix,node," [",typeof(node),"]")
end

function recursive_print(blk)
    recursive_print(blk,"")
end

macro pp(blk)
    recursive_print(blk)
end

