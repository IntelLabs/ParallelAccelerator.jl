importall ParallelAccelerator
using DocOpt
using Images

@acc function blur(img::Array{Float32,2}, iterations::Int)
    buf = Array(Float32, size(img)...) 
    runStencil(buf, img, iterations, :oob_skip) do b, a
       b[0,0] = 
            (a[-2,-2] * 0.003  + a[-1,-2] * 0.0133 + a[0,-2] * 0.0219 + a[1,-2] * 0.0133 + a[2,-2] * 0.0030 +
             a[-2,-1] * 0.0133 + a[-1,-1] * 0.0596 + a[0,-1] * 0.0983 + a[1,-1] * 0.0596 + a[2,-1] * 0.0133 +
             a[-2, 0] * 0.0219 + a[-1, 0] * 0.0983 + a[0, 0] * 0.1621 + a[1, 0] * 0.0983 + a[2, 0] * 0.0219 +
             a[-2, 1] * 0.0133 + a[-1, 1] * 0.0596 + a[0, 1] * 0.0983 + a[1, 1] * 0.0596 + a[2, 1] * 0.0133 +
             a[-2, 2] * 0.003  + a[-1, 2] * 0.0133 + a[0, 2] * 0.0219 + a[1, 2] * 0.0133 + a[2, 2] * 0.0030)
       return a, b
    end
    return img
end

function main()
    doc = """gaussian-blur.jl

Gaussian blur image processing.

Usage:
  gaussian-blur.jl -h | --help
  gaussian-blur.jl [--img-file=<img-file>] [--iterations=<iterations>]

Options:
  -h --help                  Show this screen.
  --img-file=<img-file>      Specify a path to an input image file; defaults to 'examples/gaussian-blur/sample.jpg'.
  --iterations=<iterations>  Specify a number of iterations; defaults to 100.
"""
    arguments = docopt(doc)

    if (arguments["--img-file"] != nothing)
        img_file = arguments["--img-file"]
    else
        img_file = "examples/gaussian-blur/sample.jpg"
    end

    if (arguments["--iterations"] != nothing)
        iterations = parse(Int, arguments["--iterations"])
    else
        iterations = 100
    end

    (fname, ext) = splitext(img_file)
    out_file = string(fname, "-blur", ".jpg") 

    println("input file = ", img_file)
    println("iterations = ", iterations)
    println("output file = ", out_file)

    function gaussian_blur(input_fname, output_fname, iterations)
        local img :: Array{Float32, 2} = convert(Array, float32(imread(input_fname)))
        tic()
        blur(img, 0)    
        println("SELFPRIMED ", toq())
        tic()
        img = blur(img, iterations)    
        println("SELFTIMED ", toq())
        imwrite(uint8sc(convert(Image, img)), output_fname)
    end
    
    gaussian_blur(img_file, out_file, iterations)

end

main()
