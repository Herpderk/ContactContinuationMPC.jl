using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using LinearAlgebra
using Plots
using MuJoCo
using ContactContinuationMPC

GO2 = joinpath(@__DIR__, "../../assets/unitree_go2/scene.xml")
GO2_CC = joinpath(@__DIR__, "../../assets/unitree_go2/scene_cc.xml")

Z_OFFSET = 1.0
Z_IDX = 3

N = Int(1e2)
DT = 1e-2
EPS_FD = 1e-8

function simulate_cheetah_falling(;
    m::Model, zidx::Integer, zoffset::Real, N::Integer, dt::Real, eps_fd::Real
)::Tuple{Vector{Vector{<:Real}},Vector{<:Real},Vector{<:Real}}
    # Preallocate dynamics jacobians
    ndx = get_ndx(m)
    A = mj_zeros(ndx, ndx)
    B = mj_zeros(ndx, m.nu)

    # Preallocate jacobian-norm trajectory
    Anorm_traj = zeros(N)
    Bnorm_traj = zeros(N)

    # Preallocate state trajectory
    qtraj = [zeros(m.nq) for k in 1:N]

    # Set m time step
    m.opt.timestep = dt

    # Init state zoffset above ground
    d = init_data(m)
    reset!(m, d)
    d.qpos[zidx] += zoffset

    # Simulate
    copyto!(qtraj[1], d.qpos)
    for k in 1:(N - 1)
        # Get dynamics jacobians
        forward!(m, d)
        mjd_transitionFD(m, d, eps_fd, true, A, B, nothing, nothing)

        # Update jacobian norms
        Anorm_traj[k] = norm(A)
        Bnorm_traj[k] = norm(B)

        # Update state
        copyto!(qtraj[k + 1], d.qpos)
        step!(m, d)
    end
    return qtraj, Anorm_traj, Bnorm_traj
end

function filter_for_log!(A::AbstractArray{<:Real})
    A[.!isfinite.(A)] .= 1e-6
    A[A .<= 0] .= 1e-6
end

function main()
    m = load_model(GO2)
    qs, Anorms, Bnorms = simulate_cheetah_falling(;
        m=m, zidx=Z_IDX, zoffset=Z_OFFSET, N=N, dt=DT, eps_fd=EPS_FD
    )
    zs = [q[Z_IDX] for q in qs]
    filter_for_log!(zs)
    filter_for_log!(Anorms)
    filter_for_log!(Bnorms)

    m_cc = load_model(GO2_CC)
    qs_cc, Anorms_cc, Bnorms_cc = simulate_cheetah_falling(;
        m=m_cc, zidx=Z_IDX, zoffset=Z_OFFSET, N=N, dt=DT, eps_fd=EPS_FD
    )
    zs_cc = [q[Z_IDX] for q in qs_cc]
    filter_for_log!(zs_cc)
    filter_for_log!(Anorms_cc)
    filter_for_log!(Bnorms_cc)

    ts = 0:DT:((N - 1) * DT)

    p_heights = plot(;
        title="Height Trajectories",
        #xlabel="time (s)",
        ylabel="height (m)",
        legend=:topright,
        xticks=false,
        minorticks=true,
        grid=false,
    )
    plot!(
        p_heights,
        ts,
        zs;
        label="nominal model",
        seriestype=:line,
        markersize=2,
        markerstrokewidth=0,
        #alpha=0.6
    )
    plot!(
        p_heights,
        ts,
        zs_cc;
        label="cc model",
        seriestype=:line,
        markersize=2,
        markerstrokewidth=0,
        #alpha=0.6
    )

    p_jacs = plot(;
        title="Jacobian Trajectories",
        xlabel="time (s)",
        ylabel="Jacobian norm",
        yscale=:log,
        legend=:topleft,
        minorticks=true,
        grid=false,
    )
    plot!(
        p_jacs,
        ts,
        Anorms;
        label="nominal A",
        seriestype=:line,
        markersize=2,
        markerstrokewidth=0,
        #alpha=0.6
    )
    plot!(
        p_jacs,
        ts,
        Bnorms;
        label="nominal B",
        seriestype=:line,
        markersize=2,
        markerstrokewidth=0,
        #alpha=0.6
    )
    plot!(
        p_jacs,
        ts,
        Anorms_cc;
        label="cc A",
        seriestype=:line,
        markersize=2,
        markerstrokewidth=0,
        #alpha=0.6
    )
    plot!(
        p_jacs,
        ts,
        Bnorms_cc;
        label="cc B",
        seriestype=:line,
        markersize=2,
        markerstrokewidth=0,
        #alpha=0.6
    )
    plot(p_heights, p_jacs; layout=grid(2, 1))
end

main()
