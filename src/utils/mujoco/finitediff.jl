struct FDCache{T<:AbstractFloat}
    #d_clean::Data      # Clean nominal state
    #d_step1::Data      # State with pre-computed kinematics
    d_local::Data      # The working state for perturbations

    # Pre-allocated math buffers
    qpos_p::Vector{T}
    qpos_m::Vector{T}
    qvel_p::Vector{T}
    qvel_m::Vector{T}
    dq::Vector{T}      # Perturbation vector
    dq_tan::Vector{T}  # Resulting tangent vector

    function FDCache{T}(m::Model) where {T<:AbstractFloat}
        return new{T}(
            init_data(m), #init_data(m), init_data(m),
            zeros(T, m.nq),
            zeros(T, m.nq),
            zeros(T, m.nv),
            zeros(T, m.nv),
            zeros(T, m.nv),
            zeros(T, m.nv),
        )
    end
end

function threaded_fd!(
    m::Model,
    d::Data,
    c::AbstractVector{FDCache{T}},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T};
    ϵ::T=1e-6,
)::Nothing where {T}
    if length(c) != nthreads()
        throwdim("Number of FDCaches must equal number of threads")
    end

    nv = m.nv
    nu = m.nu
    ndx = get_ndx(m)
    total_cols = ndx + nu

    # We need a clean state for position perturbations...
    #d_clean = c[1].d_clean
    #mj_copyData(d_clean, m, d)

    # ...and a state with pre-computed kinematics for the skip trick
    #d_step1 = c[1].d_step1
    #mj_copyData(d_step1, m, d)
    #mj_step1(m, d_step1)

    # Distribute the columns across CPU threads
    @threads for i in 1:total_cols
        tid = threadid()
        cache = c[tid]
        d_local = cache.d_local

        # --- EVALUATE POSITIVE PERTURBATION (+ϵ) ---
        mj_copyData(d_local, m, d)
        if i <= nv
            # Position Perturbation: Geometry changes -> FULL STEP
            #mj_copyData(d_local, m, d_clean)
            fill!(cache.dq, 0.0);
            cache.dq[i] = ϵ
            mj_integratePos(m, d_local.qpos, cache.dq, 1.0)
            #mj_step(m, d_local)
        elseif i <= ndx
            # Velocity Perturbation: Bias forces change -> FULL STEP
            #mj_copyData(d_local, m, d_clean) # Notice we start from d_clean here now!
            d_local.qvel[i - nv] += ϵ
            #mj_step(m, d_local)
        else
            # Control Perturbation: THE SKIP TRICK WORKS HERE!
            # Controls only affect step2 accelerations.
            #mj_copyData(d_local, m, d_step1) # Start AFTER step1
            d_local.ctrl[i - ndx] += ϵ
            #= # Skip trick only works for forward euler
            if m.opt.integrator == MuJoCo.mjINT_EULER
                #mj_step2(m, d_local)
                #mj_Euler(m, d_local)
                mj_step(m, d_local)
            else
                mj_step(m, d_local)
            end =#
        end
        mj_step(m, d_local)
        copyto!(cache.qpos_p, d_local.qpos)
        copyto!(cache.qvel_p, d_local.qvel)

        # --- EVALUATE NEGATIVE PERTURBATION (-ϵ) ---
        mj_copyData(d_local, m, d)
        if i <= nv
            # Position Perturbation: Geometry changes -> FULL STEP
            #mj_copyData(d_local, m, d_clean)
            fill!(cache.dq, 0.0);
            cache.dq[i] = -ϵ
            mj_integratePos(m, d_local.qpos, cache.dq, 1.0)
            #mj_step(m, d_local)
        elseif i <= ndx
            # Velocity Perturbation: Bias forces change -> FULL STEP
            #mj_copyData(d_local, m, d_clean) # Notice we start from d_clean here now!
            d_local.qvel[i - nv] -= ϵ
            #mj_step(m, d_local)
        else
            # Control Perturbation: THE SKIP TRICK WORKS HERE!
            # Controls only affect step2 accelerations.
            #mj_copyData(d_local, m, d_step1) # Start AFTER step1
            d_local.ctrl[i - ndx] -= ϵ
            #= # Skip trick only works for forward euler
            if m.opt.integrator == MuJoCo.mjINT_EULER
                mj_step2(m, d_local)
                mj_Euler(m, d_local)
            else
                mj_step(m, d_local)
            end =#
        end
        mj_step(m, d_local)
        copyto!(cache.qpos_m, d_local.qpos)
        copyto!(cache.qvel_m, d_local.qvel)

        # --- CALCULATE CENTRAL DIFFERENCE ---
        # Differentiate position (nq) back into tangent space (nv)
        mj_differentiatePos(m, cache.dq_tan, 1.0, cache.qpos_m, cache.qpos_p)
        cache.dq_tan ./= (2 * ϵ)

        # Differentiate velocity (standard Euclidean subtraction)
        @. cache.qvel_p = (cache.qvel_p - cache.qvel_m) / (2 * ϵ)

        # --- WRITE TO JACOBIAN ---
        @views if i <= ndx
            A[1:nv, i] .= cache.dq_tan
            A[(nv + 1):ndx, i] .= cache.qvel_p
        else
            B[1:nv, i - ndx] .= cache.dq_tan
            B[(nv + 1):ndx, i - ndx] .= cache.qvel_p
        end
    end
    return nothing
end
