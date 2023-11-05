using Plots
using Crayons

stack = CrayonStack()
println(stack, "normal text")
println(push!(stack, Crayon(foreground = :red)), "in red")
println(push!(stack, Crayon(foreground = :blue)), "in blue")
println(pop!(stack), "in red again")
println(pop!(stack), "normal text")