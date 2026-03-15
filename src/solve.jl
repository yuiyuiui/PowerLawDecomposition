function solve_hankel(grid::AbstractVector{T}, f::AbstractVector{T}, nseek::Int;
                      h::Real=big"0.1",
                      γ::Real=big"1.1") where {T<:Real}
    hw = min(10, (length(grid) - 1) ÷ 2)
    L0 = BigFloat(grid[hw + 1])
    L1 = BigFloat(grid[end - hw])
    @assert L0 > 0
    nsert = floor(Int, log(L1 / L0) / log(γ))
    insert_x = [T(L0 * γ^n) for n in 0:(nsert - 1)]
    power_grid = [T(1 + n * h) for n in 0:(nsert - 1)]
    insert_f = zeros(T, nsert)
    insert_f[1] = f[hw + 1]
    for n in 2:nsert
        insert_f[n] = interpolate(grid, f, insert_x[n], LagrangePola(hw))
    end

    res = matrix_pencil(power_grid, insert_f, nseek).a
    return res .* h ./ log(γ)
end
