"""
    wynn_epsilon_core(S)

The core implementation of Wynn's Epsilon algorithm.
Input a sequence S of length n (odd), output the accelerated limit value.
"""
function wynn_epsilon_core(S::AbstractVector{T}) where {T<:Real}
    n = length(S)

    e_curr = copy(S)
    e_prev = zeros(T, n)

    for col in 1:(n - 1)
        width = n - col

        e_next_prev = copy(e_curr)

        for i in 1:width
            denom = e_curr[i + 1] - e_curr[i]
            if abs(denom) < 100 * eps(T)
                e_curr[i] = e_prev[i + 1] +
                            1 * denom / (eps(T)^2 + denom^2)
            else
                e_curr[i] = e_prev[i + 1] + 1 / denom
            end
        end

        e_prev = e_next_prev
    end

    return e_curr[1]
end