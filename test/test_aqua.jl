using Test, QuantJulia, Aqua

# Package-quality gates: method ambiguities, unbound type parameters,
# undefined exports, stale dependencies, compat bounds on every dependency,
# and type piracy. Persistent-task checking is off: it spawns a fresh Julia
# and precompiles the package again, minutes of CI for a package that
# starts no tasks.
@testset "Aqua" begin
    Aqua.test_all(QuantJulia; persistent_tasks = false)
end
