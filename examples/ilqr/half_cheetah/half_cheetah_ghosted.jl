using Pkg;
Pkg.activate(joinpath(@__DIR__, "../../.."))
using LinearAlgebra
using MuJoCo
using ContactContinuationMPC
using JLD2

USE_CC = false
HALF_CHEETAH = joinpath(@__DIR__, "../../assets/half_cheetah/half_cheetah.xml")
HALF_CHEETAH_CC = joinpath(
    @__DIR__, "../../assets/half_cheetah/half_cheetah_cc.xml"
)

init_visualiser()

function main(use_cc::Bool)
    # Get smoothed dynamics model
    if use_cc
        mbwd = load_model(HALF_CHEETAH_CC)
    else
        mbwd = load_model(HALF_CHEETAH)
    end

    # Forward model is always stiff
    mfwd = load_model(HALF_CHEETAH)
    d = init_data(mfwd)
    nx = get_nx(mfwd)
    nu = mfwd.nu

    println("Joint names:")
    joint_names = get_joint_names(mfwd)
    for name in joint_names
        println(name)
    end

    # Declare references and initial conditions
    N = 200     #   4s (timestep is 0.02s)

    xidx =
        1 +
        MuJoCo.LibMuJoCo.mj_name2id(mfwd, MuJoCo.LibMuJoCo.mjOBJ_JOINT, "rootx")
    Xref = [zeros(nx) for k in 1:N]
    for k in 1:N
        copy_data_to_state!(d, Xref[k])
        qref = get_q(d, Xref[k])
        qref[xidx] += 10.0    # Set reference position without changing height
        vref = get_v(d, Xref[k])
        vref[xidx] = 1.0     # Set reference velocity
    end

    Uref = [zeros(nu) for k in 1:(N - 1)]
    xic = zeros(nx)
    copy_data_to_state!(d, xic)

    # Declare cost function (Penalize horizontal pos, vertical pos, and pitch)
    yidx =
        1 +
        MuJoCo.LibMuJoCo.mj_name2id(mfwd, MuJoCo.LibMuJoCo.mjOBJ_JOINT, "rooty")
    zidx =
        1 +
        MuJoCo.LibMuJoCo.mj_name2id(mfwd, MuJoCo.LibMuJoCo.mjOBJ_JOINT, "rootz")
    Q = 1e-5 * Matrix(I(nx))
    Q[xidx, xidx] *= 20.0
    Q[yidx, yidx] *= 5.0
    Q[zidx, zidx] *= 2.0

    # Penalize pitch and vertical position on the terminal state
    Qf = 1e+0 * Q
    Qf[yidx, yidx] *= 500.0
    Qf[zidx, zidx] *= 100.0

    R = 1e-3 * Matrix(I(mfwd.nu))
    costfunc = QuadraticCostFunction(Q, R, Qf)

    # Declare parameters and options
    params = TrajoptParameters(mfwd, mbwd, costfunc, Xref, Uref, xic)
    opts = iLQROptions(;
        maxiter_ilqr=100,
        maxiter_ls=40,
        alpha_mul=0.8,
        tol_interp=1.0,
        tol_converge=0.5,
        margin_ls=1e-2,
        eps_fd=1e-8,
    )

    # Solve trajopt
    sol = TrajoptSolution(params)
    cache = iLQRCache(params)
    run_ilqr!(sol, cache, params, opts)

    # Visualize solution
    println("\nFinal state: $(sol.X[end])\n")
    jldsave(
        "half_cheetah_qtraj.jld2";
        traj=stack(sol.X)[1:mfwd.nq, :],
        metadata="Cheetah Run 01",
    )
    jldsave("halfcheetah_Jtraj.jld2"; traj=cache.fwd.Js, metadata="")
    #visualise!(mfwd, d; trajectories=stack(sol.X))

    # --- XML INJECTION GHOSTING ---

    num_ghosts = 10
    ghost_indices = round.(Int, range(1, length(sol.X); length=num_ghosts))
    ghost_xml_lines = String[]

    # Map MuJoCo integer geom types to XML strings
    geom_type_map = Dict(
        0 => "plane",
        1 => "hfield",
        2 => "sphere",
        3 => "capsule",
        4 => "ellipsoid",
        5 => "cylinder",
        6 => "box",
    )

    original_qpos = copy(d.qpos)

    # 1. Extract global geometry data and format into XML tags
    for (step, idx) in enumerate(ghost_indices)
        # Calculate 't' (0.0 at the first ghost, 1.0 at the final ghost)
        t = (step - 1) / max(1, num_ghosts - 1)

        # Interpolate to Green from Orange
        r = 1.0 #- 1.0 * t                # Red fades from 1.0 down to 0.0
        g_color = 0.6 - (t * 0.6)        # Green ramps up from 0.6 to 1.0
        b = 0.0 * t                      # Blue ramps up from 0.0 to 0.6
        a = 0.2 + t * 0.8                # Alpha (opacity) ramps from 20% to 100%

        # Round the values to keep the XML clean
        rgba_str = "$(round(r, digits=3)) $(round(g_color, digits=3)) $(round(b, digits=3)) $(round(a, digits=3))"

        # Set the model to the current trajectory step
        d.qpos .= sol.X[idx][1:mfwd.nq]
        MuJoCo.LibMuJoCo.mj_kinematics(mfwd, d)

        for g in 1:mfwd.ngeom
            type_int = mfwd.geom_type[g]

            if type_int != 0 && type_int != 7
                type_str = get(geom_type_map, type_int, "sphere")

                s = mfwd.geom_size[g, :]
                p = d.geom_xpos[g, :]
                mat = d.geom_xmat[g, :]

                quat = zeros(4)
                MuJoCo.LibMuJoCo.mju_mat2Quat(quat, mat)

                # Inject our dynamic rgba_str here
                line = "<geom type=\"$type_str\" size=\"$(s[1]) $(s[2]) $(s[3])\" pos=\"$(p[1]) $(p[2]) $(p[3])\" quat=\"$(quat[1]) $(quat[2]) $(quat[3]) $(quat[4])\" rgba=\"$rgba_str\" contype=\"0\" conaffinity=\"0\"/>"
                push!(ghost_xml_lines, line)
            end
        end
    end

    d.qpos .= original_qpos

    # 2. Inject the tags into the original XML
    xml_path = use_cc ? HALF_CHEETAH_CC : HALF_CHEETAH
    original_xml = read(xml_path, String)

    # Insert the ghosts right before the worldbody closes
    ghosts_str = join(ghost_xml_lines, "\n        ")
    ghosted_xml = replace(
        original_xml,
        "</worldbody>" => "\n        " * ghosts_str * "\n    </worldbody>",
    )

    # 3. Save the new model to disk and load it
    ghosted_path = joinpath(@__DIR__, "half_cheetah_ghosted.xml")
    write(ghosted_path, ghosted_xml)

    m_ghost = load_model(ghosted_path)
    d_ghost = init_data(m_ghost)

    # 4. Visualize the new model using your existing black-box visualizer
    visualise!(m_ghost, d_ghost; trajectories=stack(sol.X))
end

main(USE_CC)
